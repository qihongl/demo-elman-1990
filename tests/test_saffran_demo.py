"""Checks for the stimulus and prediction alignment used in the demonstration."""

from collections import Counter
import math

import numpy as np
import torch
import torch.nn.functional as F
import saffran_demo

from saffran_demo import (
    Config,
    ElmanNet,
    SYLLABLES,
    WORD_NAMES,
    WORDS,
    bootstrap_mean_ci,
    evaluate,
    make_stream,
    prediction_pairs,
    train_model,
)


def test_stimuli_are_balanced_syllable_streams_without_word_repeats():
    for seed in (0, 1, 2026):
        stream = make_stream(seed)
        word_order = stream.word_ids[::3]
        assert Counter(word_order) == Counter({i: 45 for i in range(4)})
        assert len(stream.tokens) == 540
        assert np.all(word_order[1:] != word_order[:-1])
        assert len(SYLLABLES) == len(set(SYLLABLES)) == 12
        assert stream.is_boundary.sum() == 179
        assert not stream.is_boundary[0]
        assert np.array_equal(np.flatnonzero(stream.is_boundary), np.arange(3, 540, 3))
        for start in range(0, 540, 3):
            word = WORD_NAMES[stream.word_ids[start]]
            assert tuple(stream.syllables[start:start + 3]) == WORDS[word]


def test_prediction_inputs_have_no_boundary_symbol_and_targets_are_next_syllables():
    stream = make_stream(8, copies_per_word=1)
    x, target = prediction_pairs(stream.tokens)
    assert x.shape == (1, len(stream.tokens) - 1, 12)
    assert target.shape == (1, len(stream.tokens) - 1)
    np.testing.assert_array_equal(x.argmax(-1)[0].numpy(), stream.tokens[:-1])
    np.testing.assert_array_equal(target[0].numpy(), stream.tokens[1:])
    assert torch.all(x.sum(-1) == 1)


def test_batch_dimension_does_not_replace_the_recurrent_time_dimension():
    torch.manual_seed(3)
    model = ElmanNet(hidden_size=7)
    tokens = np.stack([make_stream(seed, copies_per_word=1).tokens for seed in (2, 4)])
    x, _ = prediction_pairs(tokens)
    batched, _ = model(x)
    individual = torch.cat([model(row.unsqueeze(0))[0] for row in x], dim=0)
    torch.testing.assert_close(batched, individual)


def test_model_predictions_cannot_use_later_input():
    torch.manual_seed(7)
    model = ElmanNet(hidden_size=6)
    x, _ = prediction_pairs(make_stream(0, copies_per_word=1).tokens)
    changed_future = x.clone()
    changed_future[:, 5:] = changed_future[:, 5:].roll(1, dims=-1)
    original, _ = model(x)
    changed, _ = model(changed_future)
    torch.testing.assert_close(original[:, :5], changed[:, :5])
    assert not torch.allclose(original[:, 5:], changed[:, 5:])


def test_reported_error_is_aligned_to_the_incoming_target_not_the_input():
    torch.manual_seed(11)
    model = ElmanNet(hidden_size=5)
    stream = make_stream(12, copies_per_word=1)
    observed = evaluate(model, stream)
    assert np.isnan(observed[0])
    assert len(observed) == len(stream.tokens)
    hidden = None
    expected = []
    with torch.no_grad():
        for previous, incoming in zip(stream.tokens[:-1], stream.tokens[1:]):
            x = F.one_hot(torch.tensor([[previous]]), 12).float()
            logits, hidden = model(x, hidden)
            expected.append(float(-F.log_softmax(logits[0, 0], dim=-1)[incoming] / np.log(2)))
    np.testing.assert_allclose(observed[1:], expected, rtol=1e-6)


def test_bootstrap_resamples_whole_model_seeds_and_preserves_time_pairing():
    values = np.array([[0., 10., 20.], [1., 11., 21.], [2., 12., 22.]])
    mean, low, high = bootstrap_mean_ci(values, seed=42)
    np.testing.assert_allclose(mean, [1., 11., 21.])
    np.testing.assert_allclose(np.diff(low), [10., 10.])
    np.testing.assert_allclose(np.diff(high), [10., 10.])
    assert np.all(low <= mean)
    assert np.all(mean <= high)
    assert np.all(high > low)
    only_mean, no_low, no_high = bootstrap_mean_ci(values[:1])
    np.testing.assert_allclose(only_mean, values[0])
    assert no_low is no_high is None


def test_training_visits_one_stream_once_and_carries_detached_context(monkeypatch):
    """Chunking must neither replay exposure nor lose cross-chunk predictions."""
    config = Config()
    seed = 0
    expected_stream = make_stream(2**32 + seed, config.copies_per_word)
    expected_inputs, expected_targets = prediction_pairs(expected_stream.tokens)
    stream_calls, input_chunks, target_chunks = [], [], []
    incoming_hidden, outgoing_hidden, optimizer_steps = [], [], []
    original_make_stream = saffran_demo.make_stream
    original_forward = ElmanNet.forward
    original_cross_entropy = F.cross_entropy
    original_step = torch.optim.Adam.step

    def record_stream(*args, **kwargs):
        stream_calls.append((args, kwargs))
        return original_make_stream(*args, **kwargs)

    def record_forward(model, inputs, hidden=None):
        input_chunks.append(inputs.detach().clone())
        incoming_hidden.append(None if hidden is None else
                               (hidden.detach().clone(), hidden.requires_grad, hidden.grad_fn))
        logits, next_hidden = original_forward(model, inputs, hidden)
        outgoing_hidden.append(next_hidden.detach().clone())
        return logits, next_hidden

    def record_loss(logits, targets, *args, **kwargs):
        target_chunks.append(targets.detach().clone())
        return original_cross_entropy(logits, targets, *args, **kwargs)

    def record_step(optimizer, *args, **kwargs):
        optimizer_steps.append(1)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(saffran_demo, "make_stream", record_stream)
    monkeypatch.setattr(ElmanNet, "forward", record_forward)
    monkeypatch.setattr(F, "cross_entropy", record_loss)
    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        _, history = train_model(config, seed)
    finally:
        torch.set_num_threads(previous_threads)

    # One 540-syllable stream provides exactly 539 next-syllable targets.
    assert len(stream_calls) == 1
    assert sum(chunk.shape[1] for chunk in input_chunks) == 539
    torch.testing.assert_close(torch.cat(input_chunks, dim=1), expected_inputs)
    torch.testing.assert_close(torch.cat(target_chunks), expected_targets.reshape(-1))
    expected_updates = math.ceil(expected_targets.numel() / config.chunk_size)
    assert expected_updates == 34
    assert len(optimizer_steps) == len(history) == len(input_chunks) == expected_updates
    assert max(chunk.shape[1] for chunk in input_chunks) == config.chunk_size
    assert input_chunks[-1].shape[1] == 11

    assert incoming_hidden[0] is None
    for previous, incoming in zip(outgoing_hidden[:-1], incoming_hidden[1:]):
        assert incoming is not None  # no context reset at chunk/word boundaries
        hidden_values, requires_grad, grad_fn = incoming
        assert not requires_grad
        assert grad_fn is None
        torch.testing.assert_close(hidden_values, previous)


def test_held_out_evaluation_never_updates_model_weights_or_gradients():
    torch.manual_seed(4)
    model = ElmanNet(hidden_size=7)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    gradients = []
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, index + 0.25)
        gradients.append(parameter.grad.clone())
    grad_modes = []
    hook = model.register_forward_pre_hook(lambda *_: grad_modes.append(torch.is_grad_enabled()))
    try:
        first = evaluate(model, make_stream(2026, copies_per_word=1))
        second = evaluate(model, make_stream(2026, copies_per_word=1))
    finally:
        hook.remove()

    assert grad_modes == [False, False]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    np.testing.assert_array_equal(first, second)

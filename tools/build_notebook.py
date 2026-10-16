"""Generate a self-contained Colab/local notebook from the tested Python source.

The notebook embeds the plotting helper so it also works without a full clone.
Run from the repository root after changing saffran_demo.py or plotting assets.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "saffran_demo.py").read_text().split('\nif __name__ == "__main__":')[0]
source = source.replace('Path(__file__).resolve().parent / "plotting"', 'Path.cwd() / "plotting"')
plot_start = source.index("def plot_prediction_error(")
train_start = source.index("def train_model(")
helper_files = ["plotting/scripts/plot_style.py", "plotting/lu.mplstyle"]
setup = "# Bundled plotting assets: no network downloads or personal paths.\nfrom pathlib import Path\n"
for path in helper_files:
    setup += (f"Path({path!r}).parent.mkdir(parents=True, exist_ok=True)\n"
              f"Path({path!r}).write_text({(ROOT / path).read_text()!r}, encoding='utf-8')\n")

nb = nbf.v4.new_notebook(cells=[
    nbf.v4.new_markdown_cell("""# Prediction error can reveal word boundaries

An Elman recurrent network predicts the **next syllable** in a continuous stream of
Saffran, Aslin & Newport's (1996), Experiment 1A, artificial language.
It receives no spaces, word identities, boundary flags, or boundary-triggered resets.

| Word | Syllables |
|---|---|
| tupiro | tu · pi · ro |
| golabu | go · la · bu |
| bidaku | bi · da · ku |
| padoti | pa · do · ti |

The original familiarization used 45 copies of each word, without immediate word
repetitions: 180 words / 540 syllables over two minutes. This simulation uses the
same inventory and stream constraints, with **one training pass per model**:
540 observed syllables and 539 next-syllable targets. This matches the nominal
stimulus count, not the biological learning rate or infant behavioral response.

Run all cells in order. Colab usually includes the scientific dependencies;
locally, install `requirements.txt` and use your Python environment as the kernel.
This notebook is self-contained and needs no repository download in Colab.
"""),
    nbf.v4.new_markdown_cell("## 1. Stimuli and recurrent network\n\nOne-hot vectors identify the 12 syllables. Input at time *t* predicts the syllable at *t + 1*. The recurrent state runs continuously across words."),
    nbf.v4.new_code_cell(source[:train_start]),
    nbf.v4.new_code_cell("for word, syllables in WORDS.items():\n    print(f'{word:7s}: {\" · \".join(syllables)}')\npreview = make_stream(2026)\nprint('\\nModel input (first 24 syllables; no boundaries):')\nprint(' '.join(preview.syllables[:24]))"),
    nbf.v4.new_markdown_cell("## 2. Learn during one brief exposure\n\nThe loss is cross-entropy, reported in bits. Each of five independent model seeds sees one 540-syllable stream, once. Online learning uses 16 prediction pairs per update (34 updates total). Hidden state values continue across chunks; gradients are detached between chunks. There are no word-boundary resets. Training and test word orders use separate seed ranges. No syllables are replayed for another epoch."),
    nbf.v4.new_code_cell(source[train_start:plot_start]),
    nbf.v4.new_markdown_cell("## 3. Evaluate and plot model mean with uncertainty\n\nThe test word order is held out of training. Error is **−log₂ P(the incoming syllable | preceding syllables)**. A boundary-crossing error is plotted on the **first syllable of the new word**. The black line is the mean across five independently exposed models; the gray band is a pointwise 95% bootstrap confidence interval for that mean (10,000 resamples of model seeds). Each model sees only one training stream. There is no added noise or temporal smoothing. The test window is unchanged from the earlier figure. Word labels and dashed boundaries are added only for the reader."),
    nbf.v4.new_code_cell(setup, metadata={"jupyter": {"source_hidden": True}}),
    nbf.v4.new_code_cell(source[plot_start:], metadata={"jupyter": {"source_hidden": True}}),
    nbf.v4.new_code_cell("errors, test_stream, result = run(Config(), Path('results'))"),
    nbf.v4.new_code_cell("from IPython.display import Image, display\ndisplay(Image(filename='imgs/pe.png'))\nprint(Path('imgs/pe.caption.md').read_text())"),
    nbf.v4.new_markdown_cell("""## Interpretation

Within each word, the next syllable is fixed. Across word boundaries, the next word
can be any of the other three: nominal transition probability 1/3. A well-trained
predictor should therefore approach 0 bits within words and log₂(3) ≈ 1.585 bits
across boundaries. This briefly exposed model remains imperfect, so both within-word
and boundary errors vary. Finite sampled streams and model estimation cause deviations.

The model is trained only on prediction, not explicit segmentation labels.
Higher boundary error is evidence of a useful segmentation cue, not proof that a
separate explicit word-segmentation representation has been learned.

The full test stream, per-seed errors, training losses, empirical transition
probabilities, and configuration are saved under `results/`.

References: [Elman (1990), Finding Structure in Time](https://doi.org/10.1016/0364-0213(90)90002-E);
[Saffran, Aslin & Newport (1996), Statistical Learning by 8-Month-Old Infants](https://doi.org/10.1126/science.274.5294.1926).
"""),
])
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
})
nbf.write(nb, ROOT / "elman_pytorch.ipynb")

# `v1/figs/`: figures of the 2025 draft and MiniGPT-4 example images

Part of v1, the original system (2025 to April 2026). Seven files are the lab's figures for the thesis-style 2025
draft in [`../documentation/archive/legacy_paper_2025/`](../documentation/archive/legacy_paper_2025/) (matplotlib
3.7.5; the PDFs are dated 15 November 2025). The other 28 were in the repository's first commit (28 June 2025) with
the MiniGPT-4 code and show MiniGPT-4 and MiniGPT-v2 on general images (food, street scenes, art, memes). No code
uses them; the April 2026 report's figures are in [`../paper/figures/`](../paper/figures/) and the 2026 revival's in
[`../../docs/figures/`](../../docs/figures/).

## Files

The draft cites its figures by number only; the scripts it names for drawing them (`create_paper_figures.py`,
`create_architecture_diagram.py`, `create_sample_images_grid.py`) are not in the repository.

| File | What it is |
|---|---|
| [`architecture_diagram.pdf`](architecture_diagram.pdf) | draft Figure 1: the dual-model architecture (ResNet-50 diagnosis passed to MiniGPT-v2 as ground truth) |
| [`architecture_diagram.png`](architecture_diagram.png) | the same as PNG |
| [`sample_images_grid.pdf`](sample_images_grid.pdf) | draft Figure 2: two training photographs of each of the seven conditions |
| [`sample_images_grid.png`](sample_images_grid.png) | the same as PNG |
| [`per_class_metrics.png`](per_class_metrics.png) | draft Figure 3: ResNet-50 precision, recall and F1 per condition on the draft's 350-image test set |
| [`confusion_matrix.pdf`](confusion_matrix.pdf) | draft Figure 4: ResNet-50 confusion matrix on that test set (50 photographs per condition) |
| [`confusion_matrix.png`](confusion_matrix.png) | the same as PNG |
| [`overview.png`](overview.png) | MiniGPT-4 architecture (ViT and Q-Former, linear layer, Vicuna) |
| [`demo.png`](demo.png) | MiniGPT-v2 task examples (`[identify]`, `[grounding]`, `[refer]`, `[vqa]`, `[detection]`) |
| [`minigpt2_demo.png`](minigpt2_demo.png) | screenshot of the MiniGPT-v2 web demo (description with boxes) |
| [`online_demo.png`](online_demo.png) | screenshot of the MiniGPT-4 web demo |
| [`examples/ad_1.png`](examples/ad_1.png) | MiniGPT-4 chat example: an advertisement for mugs |
| [`examples/ad_2.png`](examples/ad_2.png) | an advertisement for a lamp |
| [`examples/cook_1.png`](examples/cook_1.png) | a recipe for a pictured dish |
| [`examples/cook_2.png`](examples/cook_2.png) | a recipe for ramen |
| [`examples/describe_1.png`](examples/describe_1.png) | a detailed description of a street scene |
| [`examples/describe_2.png`](examples/describe_2.png) | a detailed description of a concert |
| [`examples/fact_1.png`](examples/fact_1.png) | an introduction to a film |
| [`examples/fact_2.png`](examples/fact_2.png) | an introduction to a painting and its artist |
| [`examples/fix_1.png`](examples/fix_1.png) | what is wrong with a plant's leaf |
| [`examples/fix_2.png`](examples/fix_2.png) | why a washing machine overflows |
| [`examples/fun_1.png`](examples/fun_1.png) | why a picture is funny |
| [`examples/fun_2.png`](examples/fun_2.png) | why a meme is funny |
| [`examples/logo_1.png`](examples/logo_1.png) | an opinion on a logo design |
| [`examples/op_1.png`](examples/op_1.png) | an opinion on a pictured person's behavior |
| [`examples/op_2.png`](examples/op_2.png) | words for a boy in a picture |
| [`examples/people_1.png`](examples/people_1.png) | naming characters in a comic |
| [`examples/people_2.png`](examples/people_2.png) | naming a person in a picture |
| [`examples/rhyme_1.png`](examples/rhyme_1.png) | a poem about a photograph |
| [`examples/rhyme_2.png`](examples/rhyme_2.png) | a rap song about a photograph |
| [`examples/story_1.png`](examples/story_1.png) | a story for a child |
| [`examples/story_2.png`](examples/story_2.png) | a background story for a game character |
| [`examples/web_1.png`](examples/web_1.png) | a website written from a hand-drawn sketch |
| [`examples/wop_1.png`](examples/wop_1.png) | whether a pictured scene (a wolf and a lamb) can happen in the real world |
| [`examples/wop_2.png`](examples/wop_2.png) | whether a pictured scene (a cactus on a frozen lake) is common in the real world |

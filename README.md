# hyperbolicEmbeddings_team4
Hyperbolic embeddings repository for team 4 - Santiago Cortés, David Ricardo Valencia and Juan Manuel Gutiérrez

### Installation guide
git clone git@github.com:lacunafellow/hyperbolicEmbeddings_team4.git hyppye
cd hyppye
pip install --user .

### Usage guide: create 3-dimensional embeddings from music_info.edges and save to embedding_result.txt
hyppye --dataset music_info.edges --save_embedding embedding_result.txt --dim 3

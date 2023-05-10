docker run \
 --gpus all \
 -v $(pwd):/code \
 -v $HOME/data:/data \
 --rm -it -p 8989:8989 \
 codeword-diffusion \
 jupyter notebook \
 --no-browser \
 --ip="0.0.0.0" \
 --port 8989 \
 --allow-root \
 --notebook-dir=/data
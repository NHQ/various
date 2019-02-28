(cd public/; for raw in *.raw; do ffmpeg -loglevel panic -vcodec rawvideo -f rawvideo -pix_fmt rgba -s 28x28 -i $raw -f image2 -vcodec mjpeg $raw.jpg; done)


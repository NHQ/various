(cd public/; for digit in *.jpg; do convert $digit -resize 1000% $digit; done)

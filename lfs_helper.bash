#!/bin/bash
# Find and track large files with Git LFS

# Finding all .html files over 100 MB
find . -type f -name "*.html" -size +100M | while read file
do
  git lfs track "$file"
  echo "Tracked $file with Git LFS"
done

# Add .gitattributes to stage
git add .gitattributes
#!/usr/bin/env bash
# Download the Recruit Restaurant Visitor Forecasting dataset into data/raw/.
# Requires the Kaggle CLI and accepted competition rules:
#   pip install kaggle  &&  place kaggle.json in ~/.kaggle/
# See docs/01-data.md for the file inventory and the missing-day policy.
set -euo pipefail

DEST="$(dirname "$0")/../data/raw"
mkdir -p "$DEST"

echo "Downloading recruit-restaurant-visitor-forecasting into $DEST ..."
kaggle competitions download -c recruit-restaurant-visitor-forecasting -p "$DEST"

echo "Unzipping ..."
cd "$DEST"
unzip -o '*.zip' >/dev/null
# The competition ships nested zips for each table.
for z in *.csv.zip; do [ -e "$z" ] && unzip -o "$z" >/dev/null; done
rm -f ./*.zip

echo "Done. Files in $DEST:"
ls -1 "$DEST"

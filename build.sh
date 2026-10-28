#!/usr/bin/env bash
set -e
OPENCV_DIR=${OPENCV_DIR:-/opt/homebrew/opt/opencv}

INC=""
for d in "$OPENCV_DIR/include/opencv5" "$OPENCV_DIR/include/opencv4" "$OPENCV_DIR/include"; do
  if [ -f "$d/opencv2/opencv.hpp" ]; then INC="$d"; break; fi
done
if [ -z "$INC" ]; then echo "ОШИБКА: заголовки OpenCV не найдены в $OPENCV_DIR"; exit 1; fi
echo "Использую заголовки: $INC"

g++ -std=c++17 -O2 -Wall lab1.cpp -o lab1 \
  -I"$INC" \
  -L/opt/homebrew/lib \
  -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui \
  -Wl,-rpath,/opt/homebrew/lib

echo "Сборка завершена: ./lab1"a

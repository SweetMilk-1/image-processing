# Лабораторная работа 1. Цифровая фильтрация изображений

## Сборка и запуск (C++)

OpenCV ставится через Homebrew: `brew install opencv` (на этой машине установлена
версия 5.0.0). `build.sh` сам определяет каталог заголовков (opencv4 или opencv5).

```bash
./build.sh
./lab1 test_image.tiff results --variant 1            # только сохранение
```

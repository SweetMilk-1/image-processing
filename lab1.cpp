#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;

// ---------- утилиты ----------
static inline uchar clamp8(int v)   { return (uchar)std::min(255, std::max(0, v)); }
static inline uchar clamp8f(float v){ return (uchar)std::min(255.f, std::max(0.f, v)); }

// Общая 2D-свёртка с квадратной нечётной маской, репликация границ.
// kernel — ksize*ksize, divider — нормирующий множитель (сумма коэффициентов).
// abs_resp=true — берётся модуль отклика (для контурных операторов).
static Mat convolveOdd(const Mat& src, const float* kernel, int ksize,
                       float divider, bool abs_resp)
{
    CV_Assert(src.type() == CV_8U && (ksize & 1));
    Mat dst = Mat::zeros(src.size(), CV_8U);
    const int half = ksize / 2;
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            float acc = 0.f;
            for (int ky = -half; ky <= half; ++ky) {
                int yy = std::min(src.rows - 1, std::max(0, y + ky));
                for (int kx = -half; kx <= half; ++kx) {
                    int xx = std::min(src.cols - 1, std::max(0, x + kx));
                    acc += kernel[(ky + half) * ksize + (kx + half)] *
                           (float)src.at<uchar>(yy, xx);
                }
            }
            if (divider != 0.f) acc /= divider;
            int v = abs_resp ? (int)std::round(std::fabs(acc))
                             : (int)std::round(acc);
            dst.at<uchar>(y, x) = clamp8(v);
        }
    }
    return dst;
}

// ---------- фильтры из таблицы 1.1 ----------

// 1. Фильтр Гаусса 3x3 (маска из теоретической части: {1,4,1;4,16,4;1,4,1}, k=36).
Mat filterGauss3(const Mat& src) {
    static const float K[9] = {1,4,1, 4,16,4, 1,4,1};
    return convolveOdd(src, K, 3, 36.f, false);
}

// 2. Фильтр Гаусса 5x5 (внешнее произведение [1,4,6,4,1], сумма = 256).
Mat filterGauss5(const Mat& src) {
    static const float K[25] = {
        1,4,6,4,1, 4,16,24,16,4, 6,24,36,24,6, 4,16,24,16,4, 1,4,6,4,1
    };
    return convolveOdd(src, K, 5, 256.f, false);
}

// 3. Фильтр «Мозаика» 3x3: неперекрывающиеся блоки 3x3, всем пикселям блока
// присваивается среднее по блоку; маска сдвигается на 3.
Mat filterMosaic3(const Mat& src, int bs = 3) {
    Mat dst = src.clone();
    for (int y = 0; y < src.rows; y += bs) {
        for (int x = 0; x < src.cols; x += bs) {
            int sum = 0, cnt = 0;
            for (int dy = 0; dy < bs && y + dy < src.rows; ++dy)
                for (int dx = 0; dx < bs && x + dx < src.cols; ++dx) {
                    sum += src.at<uchar>(y + dy, x + dx); ++cnt;
                }
            uchar avg = (cnt > 0) ? (uchar)(sum / cnt) : 0;
            for (int dy = 0; dy < bs && y + dy < src.rows; ++dy)
                for (int dx = 0; dx < bs && x + dx < src.cols; ++dx)
                    dst.at<uchar>(y + dy, x + dx) = avg;
        }
    }
    return dst;
}

// 4. Фильтр апертурной коррекции 3x3 (повышение резкости, ФВЧ).
//    Маска [[-1,-1,-1],[-1,X,-1],[-1,-1,-1]], X = (int)(100/S)+8, k = X-8.
//    При S=20% -> X=13, k=5; сумма ядра = 1 (DC сохраняется).
//    Защита от переполнения: clamp результата в [0,255].
Mat filterAperture(const Mat& src, double Spercent = 20.0) {
    int X = (int)(100.0 / Spercent) + 8;   // S=20 -> 13
    float k = (float)(X - 8);              // нормировка (= 5 при S=20%)
    float K[9] = {-1,-1,-1, -1,(float)X,-1, -1,-1,-1};
    return convolveOdd(src, K, 3, k, false);
}

// 5. Медианный фильтр 3x3: в фокус ставится медиана отсортированных значений окна.
Mat filterMedian3(const Mat& src, int k = 3) {
    const int half = k / 2;
    Mat dst = Mat::zeros(src.size(), CV_8U);
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            vector<uchar> v; v.reserve(k * k);
            for (int dy = -half; dy <= half; ++dy)
                for (int dx = -half; dx <= half; ++dx) {
                    int yy = std::min(src.rows - 1, std::max(0, y + dy));
                    int xx = std::min(src.cols - 1, std::max(0, x + dx));
                    v.push_back(src.at<uchar>(yy, xx));
                }
            size_t m = v.size() / 2;
            sort(v.begin(), v.end());   // 9 элементов — сортировка дёшева и однозначна
            dst.at<uchar>(y, x) = v[m]; // медиана = средний по рангу элемент
        }
    }
    return dst;
}

// ---------- контурные операторы (таблица 1.2) ----------
// Все берут |отклик| и clamp в [0,255] — защита от переполнения.

static const float SOBEL_GY[9] = {-1,-2,-1, 0,0,0, 1,2,1}; // горизонтальные контуры
static const float SOBEL_GX[9] = {-1,0,1, -2,0,2, -1,0,1}; // вертикальные контуры
static const float SOBEL_DIAG[9] = {2,1,0, 1,0,-1, 0,-1,-2}; // наклонные контуры (45°)
static const float PREWITT_GY[9] = {-1,-1,-1, 0,0,0, 1,1,1};
static const float PREWITT_GX[9] = {-1,0,1, -1,0,1, -1,0,1};
static const float SCHARR_GX[9] = {-3,0,3, -10,0,10, -3,0,3};
static const float SCHARR_GY[9] = {-3,-10,-3, 0,0,0, 3,10,3};
static const float LAPLACE[9] = {0,-1,0, -1,4,-1, 0,-1,0}; // оператор Лапласа

// Оператор Робертса (ядро 2x2) — отдельная реализация (чётное ядро).
Mat roberts(const Mat& src, bool vertical) {
    Mat dst = Mat::zeros(src.size(), CV_8U);
    for (int y = 0; y < src.rows - 1; ++y) {
        for (int x = 0; x < src.cols - 1; ++x) {
            int a = src.at<uchar>(y, x),     b = src.at<uchar>(y, x + 1);
            int c = src.at<uchar>(y + 1, x), d = src.at<uchar>(y + 1, x + 1);
            // Gx = a-d (одна диагональ), Gy = b-c (другая).
            int g = vertical ? std::abs(b - c) : std::abs(a - d);
            dst.at<uchar>(y, x) = clamp8(g);
        }
    }
    return dst;
}

// Обобщённый оператор Собела: норма вектора-градиента sqrt(Gx^2+Gy^2).
Mat sobelGeneral(const Mat& src) {
    Mat gx = convolveOdd(src, SOBEL_GX, 3, 1.f, true);
    Mat gy = convolveOdd(src, SOBEL_GY, 3, 1.f, true);
    Mat dst = Mat::zeros(src.size(), CV_8U);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            float g = std::sqrt((float)gx.at<uchar>(y,x) * gx.at<uchar>(y,x)
                              + (float)gy.at<uchar>(y,x) * gy.at<uchar>(y,x));
            dst.at<uchar>(y, x) = clamp8f(g);
        }
    return dst;
}

// Оператор Шарра: норма sqrt(Sx^2+Sy^2).
Mat scharrGeneral(const Mat& src) {
    Mat gx = convolveOdd(src, SCHARR_GX, 3, 1.f, true);
    Mat gy = convolveOdd(src, SCHARR_GY, 3, 1.f, true);
    Mat dst = Mat::zeros(src.size(), CV_8U);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            float g = std::sqrt((float)gx.at<uchar>(y,x) * gx.at<uchar>(y,x)
                              + (float)gy.at<uchar>(y,x) * gy.at<uchar>(y,x));
            dst.at<uchar>(y, x) = clamp8f(g);
        }
    return dst;
}

// Возвращает контурный препарат по варианту из табл. 1.2.
Mat contourByVariant(const Mat& src, int variant) {
    switch (variant) {
        case 1: return convolveOdd(src, SOBEL_GY,    3, 1.f, true); // Собел горизонтальные
        case 2: return convolveOdd(src, SOBEL_GX,    3, 1.f, true); // Собел вертикальные
        case 3: return convolveOdd(src, SOBEL_DIAG,  3, 1.f, true); // Собел наклонные
        case 4: return sobelGeneral(src);                           // обобщённый Собел
        case 5: return convolveOdd(src, PREWITT_GY, 3, 1.f, true); // Превитт горизонтальные
        case 6: return convolveOdd(src, PREWITT_GX, 3, 1.f, true); // Превитт вертикальные
        case 7: return roberts(src, false);                         // Робертс горизонтальные
        case 8: return roberts(src, true);                          // Робертс вертикальные
        case 9: return scharrGeneral(src);                          // Шарр
        case 10: return convolveOdd(src, LAPLACE, 3, 1.f, true);    // Лаплас
        default: return convolveOdd(src, SOBEL_GY, 3, 1.f, true);
    }
}

// ---------- DoG и Кэнни ----------

Mat dog(const Mat& src) {
    Mat g3 = filterGauss3(src);
    Mat g5 = filterGauss5(src);
    Mat diff; cv::absdiff(g3, g5, diff);            // |G3 - G5|
    Mat out;  cv::convertScaleAbs(diff, out, 7.0, 0); // усиление x7 (с насыщением)
    return out;
}

Mat cannyEdges(const Mat& src, double t1 = 50, double t2 = 200) {
    Mat edges; cv::Canny(src, edges, t1, t2);
    return edges;
}

// ---------- утилита вывода ----------

Mat labeled(const Mat& img, const string& text) {
    Mat c; cvtColor(img, c, COLOR_GRAY2BGR);
    rectangle(c, Point(0,0), Point(c.cols, 22), Scalar(0,0,0), -1);
    putText(c, text, Point(4, 16), FONT_HERSHEY_SIMPLEX, 0.45,
            Scalar(255,255,255), 1, LINE_AA);
    return c;
}

Mat montage(const vector<Mat>& imgs, int cols) {
    int rows = (int)imgs.size() / cols + ((imgs.size() % cols) ? 1 : 0);
    int w = imgs[0].cols, h = imgs[0].rows;
    Mat grid = Mat::zeros(rows * h, cols * w, imgs[0].type());
    for (size_t i = 0; i < imgs.size(); ++i) {
        Mat cell(grid, Rect((i % cols) * w, (i / cols) * h, w, h));
        imgs[i].copyTo(cell);
    }
    return grid;
}

// ---------- main ----------

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Использование: " << argv[0]
             << " <изображение> [каталог] [--variant N] [--show]\n";
        return 1;
    }
    string inPath = argv[1];
    string outDir = "results";
    int variant = 1;
    bool show = false;
    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if (a == "--show") show = true;
        else if (a == "--variant" && i + 1 < argc) variant = atoi(argv[++i]);
        else outDir = a;
    }
    fs::create_directories(outDir);

    Mat srcColor = imread(inPath, IMREAD_COLOR);
    if (srcColor.empty()) { cerr << "Не удалось загрузить: " << inPath << "\n"; return 2; }

    Mat src; cvtColor(srcColor, src, COLOR_BGR2GRAY);
    imwrite(outDir + "/00_original.png", src);
    cout << "Загружено: " << inPath
         << " (" << src.cols << "x" << src.rows << "), вариант контура: " << variant << "\n";

    // Таблица 1.1.
    vector<pair<string,Mat>> tab;
    tab.emplace_back("Gauss 3x3",        filterGauss3(src));
    tab.emplace_back("Gauss 5x5",        filterGauss5(src));
    tab.emplace_back("Mosaic 3x3",      filterMosaic3(src));
    tab.emplace_back("Aperture corr 3x3", filterAperture(src, 20.0));
    tab.emplace_back("Median 3x3",      filterMedian3(src));
    // Сохранение результатов табл. 1.1.
    imwrite(outDir + "/01_gauss3.png",      tab[0].second);
    imwrite(outDir + "/02_gauss5.png",      tab[1].second);
    imwrite(outDir + "/03_mosaic3.png",     tab[2].second);
    imwrite(outDir + "/04_aperture.png",    tab[3].second);
    imwrite(outDir + "/05_median3.png",     tab[4].second);

    // Раздел 1.7: контуры.
    Mat gDoG    = dog(src);
    Mat gCanny  = cannyEdges(src, 50, 200);
    Mat gVar    = contourByVariant(src, variant);
    imwrite(outDir + "/06_dog.png",     gDoG);
    imwrite(outDir + "/07_canny.png",   gCanny);
    imwrite(outDir + "/08_variant.png", gVar);

    // Все 10 операторов табл. 1.2 (для полноты; в отчёт идёт выбранный вариант).
    fs::create_directories(outDir + "/contours");
    const vector<string> names = {
        "01_sobel_horiz","02_sobel_vert","03_sobel_diag","04_sobel_general",
        "05_prewitt_horiz","06_prewitt_vert","07_roberts_horiz","08_roberts_vert",
        "09_scharr","10_laplace"
    };
    for (int v = 1; v <= 10; ++v) {
        Mat c = contourByVariant(src, v);
        imwrite(outDir + "/contours/" + names[v-1] + ".png", c);
    }

    // Монтаж для просмотра на экране.
    vector<Mat> cells;
    cells.push_back(labeled(src, "Original"));
    for (auto& [n, im] : tab) cells.push_back(labeled(im, n));
    cells.push_back(labeled(gDoG,   "DoG (x7)"));
    cells.push_back(labeled(gCanny, "Canny 50/200"));
    cells.push_back(labeled(gVar,   "Variant " + to_string(variant)));
    Mat m = montage(cells, 3);
    imwrite(outDir + "/montage.png", m);
    cout << "Результаты сохранены в: " << outDir << "/\n";

    if (show) {
        imshow("Lab1 montage", m);
        imshow("Original", src);
        cout << "Нажмите любую клавишу в окне для выхода...\n";
        waitKey(0);
    }
    return 0;
}

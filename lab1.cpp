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


static inline uchar clamp8(int v) { return (uchar)std::min(255, std::max(0, v)); }
static inline uchar clamp8f(float v) { return (uchar)std::min(255.f, std::max(0.f, v)); }


static Mat convolveOdd(
    const Mat& src, 
    const float* kernel, 
    int ksize,
    float divider, 
    bool abs_resp
) {
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

Mat filterGauss3(const Mat& src) {
    static const float K[9] = {1,4,1, 4,16,4, 1,4,1};
    return convolveOdd(src, K, 3, 36.f, false);
}

Mat filterGauss5(const Mat& src) {
    static const float K[25] = {
        1,4,6,4,1, 4,16,24,16,4, 6,24,36,24,6, 4,16,24,16,4, 1,4,6,4,1
    };
    return convolveOdd(src, K, 5, 256.f, false);
}

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

//    Маска [[-1,-1,-1],[-1,X,-1],[-1,-1,-1]], X = (int)(100/S)+8, k = X-8.
//    При S=20% -> X=13, k=5; сумма ядра = 1 (DC сохраняется).
//    Защита от переполнения: clamp результата в [0,255].
Mat filterAperture(const Mat& src, double Spercent = 20.0) {
    int X = (int)(100.0 / Spercent) + 8;   // S=20 -> 13
    float k = (float)(X - 8);              // нормировка (= 5 при S=20%)
    float K[9] = {-1,-1,-1, -1,(float)X,-1, -1,-1,-1};
    return convolveOdd(src, K, 3, k, false);
}

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

// Все берут |отклик| и clamp в [0,255] — защита от переполнения.
static const float SOBEL_GY[9] = {-1,-2,-1, 0,0,0, 1,2,1}; // горизонтальные контуры
static const float SOBEL_GX[9] = {-1,0,1, -2,0,2, -1,0,1}; // вертикальные контуры
static const float SOBEL_DIAG[9] = {2,1,0, 1,0,-1, 0,-1,-2}; // наклонные контуры (45°)

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

// Возвращает контурный препарат по варианту из табл. 1.2.
Mat sobelDiag(const Mat& src) {
    return convolveOdd(src, SOBEL_DIAG,  3, 1.f, true); // Собел наклонные
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
    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        if (a == "--variant" && i + 1 < argc) variant = atoi(argv[++i]);
        else outDir = a;
    }
    fs::create_directories(outDir);

    Mat srcColor = imread(inPath, IMREAD_COLOR);
    if (srcColor.empty()) { cerr << "Не удалось загрузить: " << inPath << "\n"; return 2; }

    Mat src; cvtColor(srcColor, src, COLOR_BGR2GRAY);
    imwrite(outDir + "/00_original.png", src);
    cout << "Загружено: " << inPath
         << " (" << src.cols << "x" << src.rows << "), вариант контура: " << variant << "\n";

    vector<pair<string,Mat>> tab;
    tab.emplace_back("Gauss 3x3",        filterGauss3(src));
    tab.emplace_back("Gauss 5x5",        filterGauss5(src));
    tab.emplace_back("Mosaic 3x3",      filterMosaic3(src));
    tab.emplace_back("Aperture corr 3x3", filterAperture(src, 20.0));
    tab.emplace_back("Median 3x3",      filterMedian3(src));
    imwrite(outDir + "/01_gauss3.png",      tab[0].second);
    imwrite(outDir + "/02_gauss5.png",      tab[1].second);
    imwrite(outDir + "/03_mosaic3.png",     tab[2].second);
    imwrite(outDir + "/04_aperture.png",    tab[3].second);
    imwrite(outDir + "/05_median3.png",     tab[4].second);

    Mat gDoG    = dog(src);
    Mat gCanny  = cannyEdges(src, 50, 200);
    Mat gSobel   = sobelDiag(src);
    imwrite(outDir + "/06_dog.png",     gDoG);
    imwrite(outDir + "/07_canny.png",   gCanny);
    imwrite(outDir + "/08_sobel_diag.png", gSobel);

    return 0;
}

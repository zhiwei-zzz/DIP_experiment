#include <iostream>
#include <opencv2/opencv.hpp>
#include "main.h"

void FoldDetect(cv::Mat& mat, cv::Mat& image);
Mat Mydilate(Mat src);
Mat Myerode(Mat src);
enum adaptiveMethod { meanFilter, gaaussianFilter, medianFilter };

void MyAdaptiveThreshold(Mat& src, Mat& dst, double Maxval, int Subsize, double c, adaptiveMethod method = meanFilter) {

    Mat smooth = Mat::zeros(Size(src.cols, src.rows), CV_8UC1);
    int sub = (Subsize - 1) / 2;
    int sum, up, down, left, right;
    int cols = smooth.cols;
    int rows = smooth.rows;
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            if (i < sub) left = i;
            else left = sub;
            if (sub > cols - i - 1) right = cols - i - 1;
            else right = sub;

            if (j < sub) up = j;
            else up = sub;
            if (sub > rows - j - 1) down = rows - j - 1;
            else down = sub;
            sum = 0;
            uchar* top;

            for (int k = 0; k < up + down + 1; k++) {
                top = src.ptr<uchar>(j - up + k);
                for (int l = 0; l < left + right + 1; l++) {
                    sum += top[i - left + l];
                }              
            }

            uchar* target = smooth.ptr<uchar>(j);
            target[i] = int(sum / ((left + right + 1) * (up + down + 1)));
        }
    }

    //imshow("res", smooth);

    smooth = smooth - c;
    src.copyTo(dst);
    for (int r = 0; r < src.rows; ++r) {
        const uchar* srcptr = src.ptr<uchar>(r);
        const uchar* smoothptr = smooth.ptr<uchar>(r);
        uchar* dstptr = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c) {
            if (srcptr[c] > smoothptr[c]) {
                dstptr[c] = 0;
            }
            else
                dstptr[c] = Maxval;
        }
    }
}

int main()
{
    int canny_threshold1, canny_threshold2;
    canny_threshold1 = 60;
    canny_threshold2 = 110;

    Mat srcImg = imread("C:\\Users\\DELL\\Desktop\\DIP exp\\c++\\temp.bmp");
    if (srcImg.empty()) {
        cout << "could not load image..." << endl;
        return -1;
    }

    /*imshow("Test opencv setup", srcImg);*/

    Mat Gray = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);
    for (int i = 0; i < srcImg.rows; i++) {
        
        uchar* gray_data = Gray.ptr<uchar>(i);
        for (int j = 0; j < srcImg.cols; j++) {
            uchar* src_data = srcImg.ptr<uchar>(i, j);
            gray_data[j] = (unsigned char)(src_data[0] * 0.114 + src_data[1] * 0.587 + src_data[2] * 0.299);
            //8 bit
        }
    }
    Mat GrayThre = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);

    MyAdaptiveThreshold(Gray, GrayThre, 255, 77, 8, meanFilter);
    //imshow("sdd", GrayThre);


    Mat Lap = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);

    for (int i = 1; i < Gray.rows - 1; i++) {
        uchar* data_up = Gray.ptr<uchar>(i - 1);
        uchar* data = Gray.ptr<uchar>(i);
        uchar* data_down = Gray.ptr<uchar>(i + 1);
        uchar* lap_data = Lap.ptr<uchar>(i);
        for (int j = 1; j < Gray.cols - 1; j++) {
            if ((-6 * data[j]) + 1.5 * data_up[j] + 1.5 * data_down[j] + 1.5 * data[j - 1] + 1.5 * data[j + 1] < 0) lap_data[j] = 0;
            else if ((-6 * data[j]) + 1.5 * data_up[j] + 1.5 * data_down[j] + 1.5 * data[j - 1] + 1.5 * data[j + 1] > 255) lap_data[j] = 255;
            else lap_data[j] = (-6 * data[j]) + 1.5 * data_up[j] + 1.5 * data_down[j] + 1.5 * data[j - 1] + 1.5 * data[j + 1];
        }
    }

    Mat edge;
    Canny(Lap, edge, canny_threshold1, canny_threshold2);
    imshow("canny", edge);

    Mat dilate_kernel = Mat::ones(5, 5, CV_8UC1);
    Mat dil_er0_t = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);  
    Mat dil_ero = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);
    dil_ero = Mydilate(edge);
    dil_er0_t = Mydilate(GrayThre);
    dil_ero = Myerode(dil_ero);
    dil_er0_t = Myerode(dil_er0_t);

    FoldDetect(dil_er0_t, srcImg);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(dil_ero, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int N = 20;
    int** l = new int* [3];//动态申请二维数组l 3 * N
    for (int i = 0; i < 3; ++i) {
        l[i] = new int[N];
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 1) { l[i][j] = srcImg.cols; }
            else l[i][j] = 0.;
        }
    }

    int** h = new int* [2];//动态申请二维数组h 2 * N
    for (int i = 0; i < 2; ++i) {
        h[i] = new int[N];
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 1) { h[i][j] = srcImg.rows; }
            else h[i][j] = 0.;
        }
    }
    int** m = new int* [5];//动态申请二维数组m 5 * N
    for (int i = 0; i < 5; ++i) {
        m[i] = new int[N];
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < N; j++) {
            m[i][j] = 0.;
        }
    }

    int t, g;
    t = g = 0;
    int i, j, lmax, lmin, hmin, hmax;
    for (i = 0; i < contours.size(); i++) {
        lmax = hmax = 0;
        lmin = dil_ero.cols;
        hmin = dil_ero.rows;
        for (j = 0; j < contours[i].size(); j++) {
            if (contours[i][j].x > lmax) lmax = contours[i][j].x;
            if (contours[i][j].x < lmin) lmin = contours[i][j].x;
            if (contours[i][j].y > hmax) hmax = contours[i][j].y;
            if (contours[i][j].y < hmin) hmin = contours[i][j].y;
        }

        int flag = 0;
        uchar* data = Gray.ptr<uchar>(int((hmax + hmin) / 2));
        if ((data[int((lmin + lmax) / 2)] < 127) && ((lmax - lmin) * (hmax - hmin) > 30)) {
            int sum = 0;
            for (int i0 = 0; i0 < (hmax - hmin); i0++) {
                for (int j0 = 0; j0 < (lmax - lmin); j0++) {
                    data = Gray.ptr<uchar>((hmin + i0));
                    if (data[lmin + j0] < 100) sum += 1;
                }
            }
            if (sum > int((hmax - hmin) * (lmax - lmin) / 2)) {
                m[0][g] = lmin;
                m[1][g] = lmax;
                m[2][g] = hmin;
                m[3][g] = hmax;
                m[4][g] = contourArea(contours[i]);
                g += 1;
                drawContours(srcImg, contours, i, Scalar(255, 140, 0), 3);
                flag = 1;
            }
        }
        if (flag != 1)
        {
            if (((lmax - lmin) * (hmax - hmin)) > 1000) {
                l[0][t] = lmax;
                l[1][t] = lmin;
                h[0][t] = hmax;
                h[1][t] = hmin;
                l[2][t] = arcLength(contours[i], true) / 2;
                drawContours(srcImg, contours, i, Scalar(15, 25, 105), 2);
                t += 1;
            }
        }

    }
    for (int k = 0; k < t; k++) {
        if (l[0][k] != 0 && l[1][k] != srcImg.cols) {
            rectangle(srcImg, Rect(l[1][k] - 5, h[1][k] - 5, l[0][k] + 5 - (l[1][k] - 5), h[0][k] + 5 - (h[1][k] - 5)), (255, 0, 0), 3);
            putText(srcImg, "H:" + to_string(l[2][k]), Point(l[0][k] + 5, h[0][k] + 5), FONT_HERSHEY_COMPLEX, 1, (0, 50, 255), 2);

        }
    }
    for (int k = 0; k < g; k++) {
        rectangle(srcImg, Rect(m[0][k] - 3, m[2][k] - 3, m[1][k] + 3 - (m[0][k] - 3), m[3][k] + 3 - (m[2][k] - 3)), (255, 0, 0), 3);
        putText(srcImg, "'B:" + to_string(m[4][k]), Point(m[1][k] - 10, m[3][k] - 15), FONT_HERSHEY_COMPLEX, 1, (0, 50, 255), 2);
    }
    cout << "There are " << to_string(t + g) << "defects being detected!" << endl;

    imshow("check", srcImg);

    waitKey(0);


    return 0;
}


void FoldDetect(cv::Mat& mat, cv::Mat& image)
{
    int minLineLength, maxLineGap;
    minLineLength = 90;
    maxLineGap = 100;
    vector<Vec4i> Lines;
    HoughLinesP(mat, Lines, 1, 3.1415926 / 180, 60, minLineLength, maxLineGap);
    Mat Mask = Mat::zeros(Size(mat.cols, mat.rows), CV_8UC1);
    int a;
    a = 0;

    int t1, t2;
    static vector <Vec4i> sum;
    for (rsize_t i = 0; i < Lines.size(); i++) {
        static int begin = 1;
        Vec4i I = Lines[i];


        t1 = max(I[0], I[2]) + 30;
        t2 = min(I[0], I[2]) - 30;

        if ((min(I[0], I[2]) > t2) && (max(I[0], I[2]) < t1)) {
            a += 1;
        }
        if (a > 10) {
            if (begin == 1) {
                sum.push_back(I);

                begin = 0;
            }
            if (begin != 1)
            {
                rsize_t j = 0;
                for (j = 0; j < sum.size(); j++) {

                    Vec4i member = sum[j];
                    float distance1 = sqrt(pow((I[0] - member[0]), 2) + pow((I[1] - member[1]), 2));
                    float distance2 = sqrt(pow((I[0] - member[2]), 2) + pow((I[1] - member[3]), 2));
                    if ((distance1 < 200) || (distance2 < 200)) break;

                }
                if (j == (sum.size())) {

                    sum.push_back(I);

                }
            }
            a = 0;
        }
    }
    for (rsize_t i = 0; i < sum.size(); i++) {
        Vec4i I = sum[i];
        line(image, Point(I[0], I[1]), Point(I[2], I[3]), (30, 150, 0), 5);
    }
}

Mat Mydilate(Mat src)
{
    Mat dilated_my;
    dilated_my.create(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            //uchar minV = 255;
            uchar maxV = 0;

            //遍历周围最大像素值
            for (int yi = i - 2; yi <= i + 2; yi++)
            {
                for (int xi = j - 2; xi <= j + 2; xi++)
                {
                    if (xi < 0 || xi >= src.cols || yi < 0 || yi >= src.rows)
                    {
                        continue;
                    }
                    //minV = (std::min<uchar>)(minV, img.at<uchar>(yi, xi));
                    maxV = (std::max<uchar>)(maxV, src.at<uchar>(yi, xi));
                }
            }
            dilated_my.at<uchar>(i, j) = maxV;
        }
    }
    return dilated_my;
}

Mat Myerode(Mat src)
{
    Mat dilated_my;
    dilated_my.create(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            uchar minV = 255;

            //遍历周围最大像素值
            for (int yi = i - 2; yi <= i + 2; yi++)
            {
                for (int xi = j - 2; xi <= j + 2; xi++)
                {
                    if (xi < 0 || xi >= src.cols || yi < 0 || yi >= src.rows)
                    {
                        continue;
                    }
                    //minV = (std::min<uchar>)(minV, img.at<uchar>(yi, xi));
                    minV = (std::min<uchar>)(minV, src.at<uchar>(yi, xi));
                }
            }
            dilated_my.at<uchar>(i, j) = minV;
        }
    }
    return dilated_my;
}
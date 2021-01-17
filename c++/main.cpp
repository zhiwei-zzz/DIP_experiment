#include <iostream>
#include <opencv2/opencv.hpp>
#include "main.h"

void FoldDetect(cv::Mat& mat, cv::Mat& image);
Mat Mydilate(Mat src);
Mat Myerode(Mat src);
enum adaptiveMethod { meanFilter, gaaussianFilter, medianFilter };

void MyAdaptiveThreshold(Mat& src, Mat& dst, double Maxval, int Subsize, double c, adaptiveMethod method = meanFilter) //自适应阈值分割
{
    Mat smooth = Mat::zeros(Size(src.cols, src.rows), CV_8UC1);
    int sub = (Subsize - 1) / 2;
    int sum, up, down, left, right;
    int cols = smooth.cols;
    int rows = smooth.rows;
    for (int i = 0; i < cols; i++) {     //均值滤波
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

    smooth = smooth - c; //全局减去参数值
    src.copyTo(dst);
    for (int r = 0; r < src.rows; ++r) {  //最后根据阈值判断为0还是255
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
    int canny_threshold1, canny_threshold2;  //canny双参数阈值设定
    canny_threshold1 = 60;
    canny_threshold2 = 120;

    Mat srcImg = imread("C:\\Users\\DELL\\Desktop\\DIP exp\\c++\\temp.bmp");  //加载图片
    if (srcImg.empty()) {
        cout << "could not load image..." << endl;
        return -1;
    }
    /*imshow("Test opencv setup", srcImg);*/

    Mat Gray = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);
    for (int i = 0; i < srcImg.rows; i++) {  //转为灰度图  
        uchar* gray_data = Gray.ptr<uchar>(i);
        for (int j = 0; j < srcImg.cols; j++) {
            uchar* src_data = srcImg.ptr<uchar>(i, j);
            gray_data[j] = (unsigned char)(src_data[0] * 0.114 + src_data[1] * 0.587 + src_data[2] * 0.299);
            //8 bit
        }
    }
    Mat GrayThre = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);

    MyAdaptiveThreshold(Gray, GrayThre, 255, 77, 8, meanFilter);   //自适应阈值分割用于检测褶皱
    //imshow("sdd", GrayThre);


    Mat Lap = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);

    for (int i = 1; i < Gray.rows - 1; i++) {  //使用类拉普拉斯算子个权重为[[0, 1.5, 0], [1.5, -6, 1.5], [0, 1.5, 0]]
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
    MyCanny(Lap, edge, canny_threshold1, canny_threshold2); //调用Mycanny函数进行canny处理提取边缘
    //imshow("canny", edge);

    Mat dil_er0_t = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);
    Mat dil_ero = Mat::zeros(Size(srcImg.cols, srcImg.rows), CV_8UC1);
    dil_ero = Mydilate(edge); //进行膨胀
    dil_er0_t = Mydilate(GrayThre);
    dil_ero = Myerode(dil_ero);  //进行腐蚀
    dil_er0_t = Myerode(dil_er0_t);

    FoldDetect(dil_er0_t, srcImg);   //检测折痕

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(dil_ero, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);  //检测边框

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
    for (i = 0; i < contours.size(); i++) {         //对检测到的所有轮廓进行遍历
        lmax = hmax = 0;
        lmin = dil_ero.cols;
        hmin = dil_ero.rows;
        for (j = 0; j < contours[i].size(); j++) {      //获取当前轮廓的最小外界矩形
            if (contours[i][j].x > lmax) lmax = contours[i][j].x;
            if (contours[i][j].x < lmin) lmin = contours[i][j].x;
            if (contours[i][j].y > hmax) hmax = contours[i][j].y;
            if (contours[i][j].y < hmin) hmin = contours[i][j].y;
        }

        int flag = 0;
        uchar* data = Gray.ptr<uchar>(int((hmax + hmin) / 2));      //获取最小外接矩形的几何中心点灰度值
        if ((data[int((lmin + lmax) / 2)] < 127) && ((lmax - lmin) * (hmax - hmin) > 30)) {     //虫子判别条件1：几何中心灰度值>127，且外接矩形面积<30
            int sum = 0;
            for (int i0 = 0; i0 < (hmax - hmin); i0++) {
                for (int j0 = 0; j0 < (lmax - lmin); j0++) {
                    data = Gray.ptr<uchar>((hmin + i0));
                    if (data[lmin + j0] < 100) sum += 1;    //计算深色区域面积（阈值可调，实验条件下取值100较为理想）
                }
            }
            if (sum > int((hmax - hmin) * (lmax - lmin) / 2)) {     //虫子判别条件2：外接矩形深色区域面积大于送面积的一半（可以认为虫子，杂物一般为矩形或圆形）
                m[0][g] = lmin;
                m[1][g] = lmax;
                m[2][g] = hmin;
                m[3][g] = hmax;
                m[4][g] = contourArea(contours[i]);
                g += 1;
                drawContours(srcImg, contours, i, Scalar(255, 140, 0), 3);  //确定其外围矩形的位置并勾勒轮廓、计算面积
                flag = 1;                       //符合条件1、2则判定为虫子，计标记为1
            }
        }
        if (flag != 1)      //若判断不是虫子，则判断是否为头发
        {
            if ((((lmax - lmin) * (hmax - hmin)) > 1000) && (contourArea(contours[i]) / arcLength(contours[i], true) < 0.03)) {
                //头发判别条件1：头发一般较长且尺寸较大，设定外接矩形面积阈值为1000；头发判别条件2：周长面积比很小，设置阈值0.03
                l[0][t] = lmax;
                l[1][t] = lmin;
                h[0][t] = hmax;
                h[1][t] = hmin;
                l[2][t] = arcLength(contours[i], true) / 2;
                drawContours(srcImg, contours, i, Scalar(15, 25, 105), 2);//确定外围矩形的位置并计算长度，头发丝极细，可以认为发丝长度为周长的一半
                t += 1;
            }
        }

    }
    for (int k = 0; k < t; k++) {  //框出头发并显示长度
        if (l[0][k] != 0 && l[1][k] != srcImg.cols) {
            rectangle(srcImg, Rect(l[1][k] - 5, h[1][k] - 5, l[0][k] + 5 - (l[1][k] - 5), h[0][k] + 5 - (h[1][k] - 5)), (255, 0, 0), 3);
            putText(srcImg, "H:" + to_string(l[2][k]), Point(l[0][k] + 5, h[0][k] + 5), FONT_HERSHEY_COMPLEX, 1, (0, 50, 255), 2);

        }
    }
    for (int k = 0; k < g; k++) {   //框出异物并显示面积
        rectangle(srcImg, Rect(m[0][k] - 3, m[2][k] - 3, m[1][k] + 3 - (m[0][k] - 3), m[3][k] + 3 - (m[2][k] - 3)), (255, 0, 0), 3);
        putText(srcImg, "'B:" + to_string(m[4][k]), Point(m[1][k] - 10, m[3][k] - 15), FONT_HERSHEY_COMPLEX, 1, (0, 50, 255), 2);
    }
    cout << "There are " << to_string(t + g) << "defects being detected!" << endl;  //输出文字结果

    imshow("check", srcImg);

    waitKey(0);


    return 0;
}


void FoldDetect(cv::Mat& mat, cv::Mat& image)  //消去相近的直线
{
    int minLineLength, maxLineGap;
    minLineLength = 90;
    maxLineGap = 100;
    vector<Vec4i> Lines;
    HoughLinesP(mat, Lines, 1, 3.1415926 / 180, 60, minLineLength, maxLineGap);  //霍夫直线提取
    Mat Mask = Mat::zeros(Size(mat.cols, mat.rows), CV_8UC1);
    int a;
    a = 0;

    int t1, t2;
    static vector <Vec4i> sum;
    for (rsize_t i = 0; i < Lines.size(); i++) {  //对每一条直线进行分析
        static int begin = 1;
        Vec4i I = Lines[i]; //取第一条直线


        t1 = max(I[0], I[2]) + 30;   //考虑一能框住完整直线的矩形， 判断其中有多少条直线
        t2 = min(I[0], I[2]) - 30;

        if ((min(I[0], I[2]) > t2) && (max(I[0], I[2]) < t1)) { //若在矩形内，计数加1
            a += 1;
        }
        if (a > 10) { //若有超过十条直线，将线段入队列
            if (begin == 1) {  //推入队列前队列中无其他直线
                sum.push_back(I);

                begin = 0;
            }
            if (begin != 1)  //此时队列中已有直线，需将Lines中直线与队列中进行比较，仅有当该直线与队列中直线具有一定距离时，直线入栈
            {
                rsize_t j = 0;
                for (j = 0; j < sum.size(); j++) {

                    Vec4i member = sum[j];
                    float distance1 = sqrt(pow((I[0] - member[0]), 2) + pow((I[1] - member[1]), 2));  //进行平方和运算计算距离
                    float distance2 = sqrt(pow((I[0] - member[2]), 2) + pow((I[1] - member[3]), 2));
                    if ((distance1 < 200) || (distance2 < 200)) break; //若有一端端点间距离小于200， 即退出。

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
        line(image, Point(I[0], I[1]), Point(I[2], I[3]), (30, 150, 0), 5);//绘画队列中直线
    }
}

Mat Mydilate(Mat src)  //膨胀函数 为5*5的ones核，中心为原点
{
    Mat dilated_my;
    dilated_my.create(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i)  //遍历所有
    {
        for (int j = 0; j < src.cols; ++j)
        {
            uchar maxV = 0;
            for (int yi = i - 2; yi <= i + 2; yi++)
            {
                for (int xi = j - 2; xi <= j + 2; xi++)
                {
                    if (xi < 0 || xi >= src.cols || yi < 0 || yi >= src.rows)
                    {
                        continue;
                    }
                    maxV = (std::max<uchar>)(maxV, src.at<uchar>(yi, xi)); //取最大值
                }
            }
            dilated_my.at<uchar>(i, j) = maxV;
        }
    }
    return dilated_my;
}

Mat Myerode(Mat src)  //腐蚀函数 为5*5的ones核，中心为原点
{
    Mat dilated_my;
    dilated_my.create(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i) //遍历所有
    {
        for (int j = 0; j < src.cols; ++j)
        {
            uchar minV = 255;
            for (int yi = i - 2; yi <= i + 2; yi++)
            {
                for (int xi = j - 2; xi <= j + 2; xi++)
                {
                    if (xi < 0 || xi >= src.cols || yi < 0 || yi >= src.rows)
                    {
                        continue;
                    }

                    minV = (std::min<uchar>)(minV, src.at<uchar>(yi, xi));  //取最小值
                }
            }
            dilated_my.at<uchar>(i, j) = minV;
        }
    }
    return dilated_my;
}
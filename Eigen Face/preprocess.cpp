//#include <iostream>
//#include <fstream>
//#include <string>
//#include <opencv2/opencv.hpp>
//
//
//using namespace std;
//using namespace cv;
//
//int ROIX = 0;
//int ROIY = 0;
//Point lefteye;
//Point righteye;
//Rect ROI;
//Mat pureImg;
//int countIndex = 1;
//const int ROIWIDTH = 92;
//const int ROIHEIGHT = 112;
//const string myFaceDir = "att-face/s41";
//const int NUMPERPERSON = 20;
//
//void showImageWithMask(Mat img, string windowName = "test");
//void saveROI(Mat img);
//void setROI(Mat img);
//void processImg(Mat img);
//void toGray();
//
//
//int main(int argc, char* argv[]) {
//
//    //原图958 x 1280
//    //92 x 112 rectangle
//
//    //Mat my_pic;
//    //Mat my_pic_resized;
//    //for (int i = 1; i <= NUMPERPERSON; i++)
//    //{
//    //    string path = "att-face/s41/" + to_string(i) + ".jpg";
//    //    my_pic = imread(path);
//    //    resize(my_pic, my_pic_resized, Size(150, 128));
//    //    pureImg = my_pic_resized.clone();
//    //    processImg(my_pic_resized);
//    //}
//    toGray(); //转换为灰度图
//}
//void toGray() {
//    Mat my_pic;
//    for (int i = 1; i <= 10; i++) {
//        string path = "att-face/s41" + to_string(i) + "_roi.jpg";
//        my_pic = imread(path);
//        Mat res;
//        cvtColor(my_pic, res, COLOR_BGR2GRAY);
//        imwrite("att-face/s41" + to_string(i) + ".pgm", res);
//    }
//
//}
//void processImg(Mat img) {
//    Mat i;
//    img.copyTo(i);
//    showImageWithMask(i);
//    Mat img2;
//    int t = waitKey(-1);
//    switch (t) {
//
//        //save:q
//    case 113:
//        cout << "save roi";
//        saveROI(i);
//        break;
//        //+
//    case 61:
//        resize(img, img2, Size(img.cols * 1.1, img.rows * 1.1), 0, 0, INTER_LINEAR);
//        ROIX = img2.cols / 2 - ROIWIDTH / 2;
//        ROIY = img2.rows / 2 - ROIHEIGHT / 2;
//        destroyWindow("test");
//        processImg(img2);
//        break;
//
//        //-
//    case 45:
//        resize(img, img2, Size(img.cols * 0.9, img.rows * 0.9), 0, 0, INTER_LINEAR);
//        ROIX = img2.cols / 2 - ROIWIDTH / 2;
//        ROIY = img2.rows / 2 - ROIHEIGHT / 2;
//        destroyWindow("test");
//        processImg(img2);
//        break;
//
//        //key right
//    case 100:
//        ROIX++;
//        destroyWindow("test");
//        processImg(img);
//        break;
//        //key left
//    case 97:
//        ROIX--;
//        destroyWindow("test");
//        processImg(img);
//        break;
//        //key down
//    case 115:
//        ROIY--;
//        destroyWindow("test");
//        processImg(img);
//        break;
//        //key up
//    case 119:
//        ROIY++;
//        destroyWindow("test");
//        processImg(img);
//        break;
//    default:
//        cout << "no matching key input";
//        destroyWindow("test");
//        processImg(img);
//        break;
//
//    }
//}
//
//void saveROI(Mat img) {
//    Mat imgROI = Mat(pureImg, ROI);
//    namedWindow("imgROI");
//    imshow("imgROI", imgROI);
//    waitKey();
//    //cout << myFaceDir + "1_roi.jpg";
//    imwrite(myFaceDir + to_string(countIndex) + "_roi.jpg", imgROI);   //pgm需要灰度图才能写--到时候再改下
//    countIndex++;
//
//}
//
//void setROI(Mat img) {
//    lefteye = Point(ROIX + ROIWIDTH / 3 - 5, ROIY + ROIHEIGHT / 2);
//    righteye = Point(ROIX + ROIWIDTH * 2 / 3 + 5, ROIY + ROIHEIGHT / 2);
//    ROI = Rect(ROIX, ROIY, ROIWIDTH, ROIHEIGHT);
//}
//void showImageWithMask(Mat img, string windowName) {
//    setROI(img);
//    rectangle(img, ROI, Scalar(0, 0, 255));
//    line(img, lefteye - Point(5, 0), lefteye + Point(5, 0), Scalar(0, 0, 255));
//    line(img, lefteye - Point(0, 5), lefteye + Point(0, 5), Scalar(0, 0, 255));
//    line(img, righteye - Point(5, 0), righteye + Point(5, 0), Scalar(0, 0, 255));
//    line(img, righteye - Point(0, 5), righteye + Point(0, 5), Scalar(0, 0, 255));
//    namedWindow(windowName);
//    imshow(windowName, img);
//    //waitKey();
//}
//

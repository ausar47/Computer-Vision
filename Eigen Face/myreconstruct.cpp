//#include <iostream>
//#include <fstream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
//
//using namespace cv;
//using namespace std;
//
//class Face;
//class Person;
//
//class Face {
//public:
//    int id;
//    Mat img;
//    Person* person;
//    Mat coordinates;
//    explicit Face(Person* p, int id) : person(p), id(id) {}
//};
//
//class Person {
//public:
//    int id;
//    vector<Face*> faces;
//    explicit Person(int id) : id(id) {}
//};
//
//vector<Face*> allFaces;
//
//Size standardSize = Size(25, 25);
//Mat A;
//int eigenCount;
//String model;
//String testImgPath;
//Mat testImg;
//
//void readFaces();
//void reconstruct();
//
//int main(int argc, char* argv[]) {
//    model = String(argv[2]);
//    testImgPath = String(argv[1]);
//    readFaces();
//    reconstruct();
//    waitKey(0);
//    return 0;
//}
//
//void readFaces() {
//    for (int i = 1; i <= 40; i++) {
//        Person* person = new Person(i);
//        String fileName;
//        for (int j = 1; j <= 10; j++) {
//            Face* face = new Face(person, j);
//            fileName = "att-face/s" + to_string(i) + "/" + to_string(j) + ".pgm";
//            Mat img = imread(fileName, IMREAD_GRAYSCALE);
//            resize(img, img, standardSize);
//            face->img = Mat(standardSize.height, standardSize.width, CV_8UC1);
//            img.copyTo(face->img);
//            normalize(face->img, face->img, 255, 0, NORM_MINMAX);
//            person->faces.push_back(face);
//            allFaces.push_back(face);
//        }
//    }
//    testImg = imread(testImgPath, IMREAD_GRAYSCALE);
//    resize(testImg, testImg, standardSize);
//    normalize(testImg, testImg, 255, 0, NORM_MINMAX);
//    ifstream in(model + "/modelfile");
//    vector<String> fileNames;
//    int width, height, num;
//    in >> width >> height;
//    float energyPercent;
//    in >> energyPercent;
//    in >> num;
//    for (int i = 0; i < num; i++) {
//        string fileName;
//        in >> fileName;
//        fileNames.push_back(fileName);
//    }
//    in >> eigenCount;
//    A = Mat(standardSize.height * standardSize.width, eigenCount, CV_64F);
//    int row, col;
//    in >> row >> col;
//    for (int i = 0; i < row; i++) {
//        for (int j = 0; j < col; j++) {
//            in >> A.at<double>(i, j);
//        }
//    }
//}
//
//void reconstruct() {
//    Mat testDoubleMat;
//    testImg.reshape(0, 1).convertTo(testDoubleMat, CV_64F);
//    Mat mapped = testDoubleMat * A;
//    Mat AT = Mat(Size(A.rows, A.cols), CV_64F);
//    transpose(A, AT);
//    Mat f_hat = mapped * AT;
//    f_hat.reshape(0, standardSize.height).convertTo(f_hat, CV_64F);
//    imwrite(model+ "/reconstruction.jpg", f_hat);
//}
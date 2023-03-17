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
//vector<Person*> persons;
//vector<Face*> allFaces;
//vector<Mat> allFaceMats;
//vector<String> fileNames;
//
//Size standardSize = Size(25, 25);
//Mat covarMat = Mat();
//Mat meanMat = Mat();
//Mat eigenVectors = Mat();
//Mat eigenValues = Mat();
//vector<Mat> eigenFaces;
//Mat AT;
//Mat A;
//double energyPercent;
//int eigenCount;
//String model;
//
//void readFaces();
//void calcEigens();
//
//int main(int argc, char* argv[]) {
//    energyPercent = atof(argv[1]);
//    model = String(argv[2]);
//    readFaces();
//    calcEigens();
//    waitKey(0);
//    return 0;
//}
//
//void readFaces() {
//    for (int i = 1; i <= 41; i++) {
//        Person* person = new Person(i);
//        persons.push_back(person);
//        String fileName;
//        for (int j = 1; j <= 10; j++) {
//            Face* face = new Face(person, j);
//            fileName = "att-face/s" + to_string(i) + "/" + to_string(j) + ".pgm";
//            fileNames.push_back(fileName);
//            Mat img = imread(fileName, IMREAD_GRAYSCALE);
//            resize(img, img, standardSize);
//            face->img = Mat(standardSize.height, standardSize.width, CV_8UC1);
//            img.copyTo(face->img);
//            normalize(face->img, face->img, 255, 0, NORM_MINMAX);
//            allFaceMats.push_back(face->img);
//            person->faces.push_back(face);
//            allFaces.push_back(face);
//        }
//    }
//}
//
//void calcEigens() {
//    calcCovarMatrix(allFaceMats, covarMat, meanMat, COVAR_NORMAL);
//    Mat meanImg;
//    meanMat.convertTo(meanImg, CV_8UC1);
//    imwrite(model + "/mean.pgm", meanImg);
//    eigen(covarMat, eigenValues, eigenVectors);
//    eigenCount = eigenVectors.rows * energyPercent;
//    cout << "eigen count: " << eigenCount << endl;
//    AT = Mat(eigenCount, standardSize.height * standardSize.width, CV_64F);
//    A = Mat(standardSize.height * standardSize.width, eigenCount, CV_64F);
//    for (int i = 0; i < eigenCount; i++) {
//        Mat t = Mat(standardSize.height, standardSize.width, CV_64F);
//        Mat tt = Mat(standardSize.height, standardSize.width, CV_8UC1);
//        for (int j = 0; j < standardSize.width * standardSize.height; j++) {
//            t.at<double>(1.0 * j / standardSize.width, j % standardSize.width) = eigenVectors.at<double>(i, j);
//            AT.at<double>(i, j) = eigenVectors.at<double>(i, j);
//            A.at<double>(j, i) = eigenVectors.at<double>(i, j);
//        }
//        normalize(t, t, 255, 0, NORM_MINMAX);
//        t.convertTo(tt, CV_8UC1);
//        eigenFaces.push_back(tt);
//    }
//    string modelfile = model + "/modelfile";
//    ofstream out(modelfile);
//    out << standardSize.width << ' ' << standardSize.height << '\n';
//    out << energyPercent << '\n';
//    out << fileNames.size() << '\n';
//    for (const auto& it : fileNames) {
//        out << it << '\n';
//    }
//    out << eigenCount << '\n';
//    out << A.rows << ' ' << A.cols << '\n';
//    for (int i = 0; i < A.rows; i++) {
//        for (int j = 0; j < A.cols; j++) {
//            out << A.at<double>(i, j) << ' ';
//        }
//    }
//    out << '\n';
//    Mat eigenFacesConnected;
//    vector<Mat> teneigenFaces;
//    for (int i = 0; i < 10 && i < eigenCount; i++) {
//        teneigenFaces.push_back(eigenFaces.at(i));
//    }
//    hconcat(teneigenFaces, eigenFacesConnected);
//    imwrite(model + "/eigen faces.pgm", eigenFacesConnected);
//}
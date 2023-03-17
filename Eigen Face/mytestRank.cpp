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
//float energyPercent;
//int eigenCount;
//String model;
//String testImgPath;
//Mat testImg;
//int cate_count[41] = { 0 };
//
//void recognize(string testFileName, int index, int cate_index);
//void readFaces();
//void calcEigens();
//
//int main(int argc, char* argv[]) {
//    energyPercent = atof(argv[1]);
//    testImgPath = String(argv[2]);
//    readFaces();
//    calcEigens();
//    string fileName;
//    for (int i = 1; i <= 40; i++) {
//        for (int j = 6; j <= 10; j++)    //6-10 pic
//        {
//            fileName = testImgPath + 's' + to_string(i) + "/" + to_string(j) + ".pgm";
//            cout << fileName << endl;
//            recognize(fileName, j, i);
//        }
//    }
//    for (int i = 1; i <= 40; i++) {
//        double rank_1_rate = cate_count[i] / 5.0;
//        cout << "category " + to_string(i) + "ranke-1 rate: " << rank_1_rate << endl;
//    }
//    waitKey(0);
//    return 0;
//}
//
//void readFaces() {
//    for (int i = 1; i <= 40; i++) {
//        Person* person = new Person(i);
//        persons.push_back(person);
//        String fileName;
//        for (int j = 1; j <= 5; j++) {
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
//            t.at<double>(j / standardSize.width, j % standardSize.width) = eigenVectors.at<double>(i, j);
//            AT.at<double>(i, j) = eigenVectors.at<double>(i, j);
//            A.at<double>(j, i) = eigenVectors.at<double>(i, j);
//        }
//        normalize(t, t, 255, 0, NORM_MINMAX);
//        t.convertTo(tt, CV_8UC1);
//        eigenFaces.push_back(tt);
//    }
//}
//
//void recognize(string testFileName, int index, int cate_index) {
//    testImg = imread(testFileName, IMREAD_GRAYSCALE);
//    resize(testImg, testImg, standardSize);
//    normalize(testImg, testImg, 255, 0, NORM_MINMAX);
//    Mat testDoubleMat;
//    testImg.reshape(0, 1).convertTo(testDoubleMat, CV_64F);
//    Mat testCoordinates = testDoubleMat * A;
//    double minDistance = -1;
//    Face* resultFace = NULL;
//    for (vector<Face*>::iterator iter = allFaces.begin(); iter != allFaces.end(); iter++) {
//        double distance;
//        Face* face = *iter;
//        Mat doubleMat;
//        face->img.reshape(0, 1).convertTo(doubleMat, CV_64F);
//        face->coordinates = doubleMat * A;
//        distance = 0;
//        for (int i = 0; i < eigenCount; i++) {
//            distance += pow(face->coordinates.at<double>(0, i) - testCoordinates.at<double>(0, i), 2);
//        }
//        if (distance < minDistance || minDistance == -1) {
//            minDistance = distance;
//            resultFace = face;
//        }
//    }
//
//    cout << "test_category is: " << cate_index << endl;
//    cout << "find_category is: " << resultFace->person->id << endl;
//
//    if (cate_index == resultFace->person->id) {
//        cate_count[cate_index]++;
//    }
//}
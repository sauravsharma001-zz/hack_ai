import numpy as np
from numpy import random
import cv2
import os
from sys import argv
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


# declaring variables
TRAINING_DATA_IMG_PATH = "input/align_cropped/"


def load_data_set(feat_detect):
    """
    creating training and test set containing lists of feature extracted from individual image
    :param feat_detect: feature detector to use
    :return: training and test data set containing list of features
    """
    data = []

    print("Loading Data .....")
    folder_list = os.listdir(TRAINING_DATA_IMG_PATH)
    for folder in folder_list:
        file_list = os.listdir(TRAINING_DATA_IMG_PATH + folder + "/")
        for image_name in file_list:
            img = cv2.imread(TRAINING_DATA_IMG_PATH + folder + "/" + image_name)
            (kp, desc) = get_features(img, feat_detect)
            data.append((desc, folder))

    random.shuffle(data)
    partition = int(0.7 * np.shape(data)[0])
    training_data, test_data = data[:partition], data[partition:]

    return np.array(training_data), np.array(test_data)


def get_features(image, feature_detector):
    """
    extract features from image given a feature detector
    :param image: given image
    :param feature_detector: feature detector to use
    :return: list of key points and features
    """

    width = 256
    gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = gs_image.shape[:2]
    new_h = int((width * h) / w)
    gs_image = cv2.resize(gs_image, (width, new_h))
    kp, descriptors = feature_detector.detectAndCompute(gs_image, mask=None)
    if np.shape(kp)[0] == 0:
        return None, None
    if descriptors is None:
        return kp, None
    return kp, np.array(descriptors)


def initializing_classifier(clust_cnt):
    """
    initializing k-means and other classifiers
    :param clust_cnt: # of cluster
    :return: all classifiers
    """
    knn_classifier = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='brute')
    svm_classifier = SVC(probability=True, kernel='linear', C=1, gamma=5.383)
    ada_classifier = SVC(probability=True, kernel='linear', C=3.67)

    # ada_classifier = AdaBoostClassifier(SVC(probability=True, kernel='linear', C=3.67, gamma=5.383), n_estimators=100,
    #                                     learning_rate=1.0, algorithm='SAMME')

    kmeans_classifier = KMeans(clust_cnt)
    feature_detector = cv2.xfeatures2d.SIFT_create()
    return knn_classifier, svm_classifier, ada_classifier, kmeans_classifier, feature_detector


def k_mean_clustering(descriptor_list, k_means):
    """
    clustering feature of training set
    :param descriptor_list: list of feature
    :param k_means: K Means Classifier
    :return:
    """
    descriptors = descriptor_list[0][0]
    for descriptor, label in descriptor_list[1:]:
        if descriptor is not None:
            descriptors = np.vstack((descriptors, descriptor))

    k_means.fit(descriptors)
    return k_means


def train_classifier(knn_classifier, svm_classifier, ada_classifier, train_data, train_label):
    """
    training all the classifiers
    :param knn_classifier: KNN Classifier
    :param svm_classifier: SVM Classifier
    :param ada_classifier: AdaBoost Classifier
    :param train_data: Training Set Feature
    :param train_label: Training Set Target Variable
    :return: trained classifiers
    """
    # print('Training SVM with AdaBoost Classifier')
    # ada_classifier.fit(train_data, train_label)
    print('Training KNN Classifier')
    knn_classifier.fit(train_data, train_label)
    print('Training SVM Classifier')
    svm_classifier.fit(train_data, train_label)
    return knn_classifier, svm_classifier, ada_classifier


def bag_of_features(descriptor_list, k_mean_cluster, k_clusters):
    """
    creating bag of features for the training data using the k mean classifier result
    :param descriptor_list: training set containing descriptor for the training images
    :param k_mean_cluster: trained K Means cluster classifier
    :param k_clusters: no of cluster
    :return:
    """
    no_of_data = np.shape(descriptor_list)[0]

    x_lab = np.zeros((no_of_data, k_clusters))
    y_lab = descriptor_list[:, -1]
    t = 0
    for i in range(no_of_data):
        d = descriptor_list[i][0]
        if d is not None:
            for j in range(np.shape(d)[0]):
                cluster_index = k_mean_cluster[t]
                x_lab[i][cluster_index] = x_lab[i][cluster_index] + 1
                t = t + 1

    return x_lab, y_lab


def predict_accuracy(knn_classifier, svm_classifier, ada_classifier, k_means, test_set, k_cluster_no):
    """
    Print the accuracy of different given classifiers with the test data
    :param knn_classifier: trained KNN Classifier
    :param svm_classifier: trained SVM Classifier
    :param ada_classifier: trained AdaBoost Classifier
    :param k_means: trained K Means Classifier
    :param test_set: test set
    :param k_cluster_no: # of cluster
    """
    test_feature = np.zeros((np.shape(test_set)[0], k_cluster_no))
    test_label = test_set[:, -1]
    for i in range(np.shape(test_set)[0]):
        desc, label = test_set[i][0], test_set[i][1]
        if desc is not None:
            r = k_means.predict(desc)
            r_unique = np.unique(r, return_counts=True)
            for j in range(np.shape(r_unique)[1]):
                test_feature[i][r_unique[0][j]] = r_unique[1][j]

    knn_result = knn_classifier.predict(test_feature)
    svm_result2 = svm_classifier.predict(test_feature)
    # ada_result3 = ada_classifier.predict(test_feature)

    knn_acc = svm_acc = ada_acc = 0
    for l in range(np.shape(test_feature)[0]):
        if test_label[l] == knn_result[l]:
            knn_acc = knn_acc + 1
        if test_label[l] == svm_result2[l]:
           svm_acc = svm_acc + 1
        # if test_label[l] == ada_result3[l]:
        #     ada_acc = ada_acc + 1

    knn_acc = (knn_acc / np.shape(test_feature)[0]) * 100
    svm_acc = (svm_acc / np.shape(test_feature)[0]) * 100
    ada_acc = (ada_acc / np.shape(test_feature)[0]) * 100
    print('KNN: ', knn_acc, '%; SVM: ', svm_acc, '%, ADA: ', ada_acc, '%')


if __name__ == "__main__":

    if len(argv) == 1:
        k_cluster = 30
        print("Initializing Classifiers .....")
        knn_clr, svm_clr, ada_clr, k_means, fd = initializing_classifier(k_cluster)
        training_set, test_set = load_data_set(fd)

        print('Clustering features into', k_cluster, 'clusters .....')
        k_mean_clr = k_mean_clustering(training_set, k_means)
        k_mean_filename = 'finalized_model_k_mean.sav'
        pickle.dump(k_mean_clr, open(k_mean_filename, 'wb'))

        print('Creating Bag of Features .....')
        x_label, y_label = bag_of_features(training_set, k_mean_clr.labels_, k_cluster)

        clf, svm_clf, ada_clf = train_classifier(knn_clr, svm_clr, ada_clr, x_label, y_label)

        svm_filename = 'finalized_model_svm.sav'
        knn_filename = 'finalized_model_knn.sav'
        pickle.dump(svm_clf, open(svm_filename, 'wb'))
        pickle.dump(clf, open(knn_filename, 'wb'))

        predict_accuracy(clf, svm_clf, ada_clf, k_mean_clr, test_set, k_cluster)
import math
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from typing import Tuple, Union
from sklearn.model_selection import train_test_split


class RBFNN(BaseEstimator):
    def __init__(
        self,
        classification: bool,
        ratio_rbf: float,
        l2: bool,
        eta: float,
        logisticcv: bool,
        random_state: int,
    ) -> None:
        """
        Constructor of the class

        Parameters
        ----------
        classification: bool
            True if it is a classification problem
        ratio_rbf: float
            Ratio (as a fraction of 1) indicating the number of RBFs
            with respect to the total number of patterns
        l2: float
            True if we want to use L2 regularization for logistic regression
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
        logisticcv: bool
            True if we want to use LogisticRegressionCV
        random_state: int
            Seed for the random number generator
        """

        self.is_fitted = False
        self.classification = classification
        self.ratio_rbf = ratio_rbf
        self.l2 = l2
        self.eta = eta
        self.logisticcv = logisticcv
        self.random_state = random_state
        self.num_rbf = None
        self.centroids = None
        self.radii = None
        self.coefficients = None
        self.logreg = None

    def fit(self, X: np.array, y: np.array):
        """
        Fits the model to the input data

        Parameters
        ----------
        X: array, shape (n_patterns,n_inputs)
            Matrix with the inputs for the patterns to fit
        y: array, shape (n_patterns,n_outputs)
            Matrix with the outputs for the patterns to fit

        Returns
        -------
        self: object
            Returns an instance of self
        """

        np.random.seed(self.random_state)
        self.num_rbf = max(1, int(self.ratio_rbf * X.shape[0]))
        print(f"Number of RBFs used: {self.num_rbf}")

        # 1. Init centroids
        if self.classification:
            self.centroids = self._init_centroids_classification(X, y)
        else:
            self.centroids = X[np.random.choice(X.shape[0], self.num_rbf, replace=False)]

        # 2. clustering
        kmeans = self._clustering(X, y)
        self.centroids = kmeans.cluster_centers_

        # 3. Calculate radii
        self.radii = self._calculate_radii()

        # 4. R matrix
        distances = cdist(X, self.centroids)
        r_matrix = self._calculate_r_matrix(distances)

        # 5. Calculate betas
        if self.classification:
            self.logreg = self._logreg_classification()
            self.logreg.fit(r_matrix, y)
        else:
            self.coefficients = self._invert_matrix_regression(r_matrix, y)

        self.is_fitted = True

        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the output of the model for a given input

        Parameters
        ----------
        X: array, shape (n_patterns,n_inputs)
            Matrix with the inputs for the patterns to predict

        Returns
        -------
        predictions: array, shape (n_patterns,n_outputs)
            Predictions for the patterns in the input matrix
        """

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        # 2. clustering
        distances = cdist(X, self.centroids)
        r_matrix = self._calculate_r_matrix(distances)

        # 4. radii for test set
        self.radii = self._calculate_radii()

        if self.classification:
            predictions = self.logreg.predict(r_matrix)
        else:
            predictions = r_matrix @ self.coefficients
            # Redondear las predicciones para simular clasificación
            #predictions = np.round(predictions)

        return predictions

    def score(self, X: np.array, y: np.array):
        """
        Returns the score of the model for a given input and output

        Parameters
        ----------
        X: array, shape (n_patterns,n_inputs)
            Matrix with the inputs for the patterns to predict
        y: array, shape (n_patterns,n_outputs)
            Matrix with the outputs for the patterns to predict

        Returns
        -------
        score: float
            Score of the model for the given input and output. It can be
            accuracy or mean squared error depending on the classification
            parameter
        """
        predictions = self.predict(X)

        if self.classification:
            return accuracy_score(y, predictions)

        else:
            return mean_squared_error(y, predictions)

    def _init_centroids_classification(
        self, X_train: np.array, y_train: np.array
    ) -> np.array:
        """
        Initialize the centroids for the case of classification

        This method selects num_rbf patterns in a stratified manner.


        Parameters
        ----------
        X_train: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        y_train: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset

        Returns
        -------
        centroids: array, shape (num_rbf)
            Array with the centroids selected
        """

        classes = np.unique(y_train)
        class_share = math.floor(self.num_rbf/len(classes))
        remaining = self.num_rbf - (class_share * len(classes))

        centroids = []
        for cls in classes:
            cls_indices = np.where(y_train == cls)[0]
            if remaining > 0:
                selected = train_test_split(cls_indices, test_size=class_share+1, stratify=y_train[cls_indices])[1]
                remaining -= 1
            else:
                selected = train_test_split(cls_indices, test_size=class_share, stratify=y_train[cls_indices])[1]
            centroids.extend(X_train[selected])
        
        return centroids

    def _clustering(self, X_train: np.array, y_train: np.array) -> Tuple[KMeans]:
        """
        Apply the clustering process

        A clustering process is applied to set the centers of the RBFs.
        In the case of classification, the initial centroids are set
        using the method init_centroids_classification().
        In the case of regression, the centroids have to be set randomly.

        Parameters
        ----------
        X_train: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        y_train: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset

        Returns
        -------
        kmeans: sklearn.cluster.KMeans
            KMeans object after the clustering
        """
        kmeans = KMeans(
            n_clusters=self.num_rbf,
            init=self.centroids,
            n_init=1,
            max_iter=500,
            random_state=self.random_state,
            )
        kmeans.fit(X_train)
        
        return kmeans

    def _calculate_radii(self) -> np.array:
        """
        Obtain the value of the radii after clustering

        This methods is used to heuristically obtain the radii of the RBFs
        based on the centers

        Returns
        -------
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
        """
        dist_matrix = squareform(pdist(self.centroids))
        n_centroids = dist_matrix.shape[0]
        radii = np.sum(dist_matrix, axis=1)/(2 * (n_centroids - 1))
        
        return radii

    def _calculate_r_matrix(self, distances: np.array) -> np.array:
        """
        Obtain the R matrix

        This method obtains the R matrix (as explained in the slides),
        which contains the activation of each RBF for each pattern, including
        a final column with ones, to simulate bias

        Parameters
        ----------
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center

        Returns
        -------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        """
        activations = np.exp(- (distances ** 2) / (2 * self.radii[np.newaxis, :] ** 2))
        r_matrix = np.hstack([activations, np.ones((activations.shape[0], 1))])

        return r_matrix

    def _invert_matrix_regression(
        self, r_matrix: np.array, y_train: np.array
    ) -> np.array:
        """
        Invert the matrix for regression case

        This method obtains the pseudoinverse of the r matrix and multiplies
        it by the targets to obtain the coefficients in the case of linear
        regression

        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        y_train: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset

        Returns
        -------
        coefficients: array, shape (n_outputs,num_rbf+1)
            For every output, values of the coefficients for each RBF and value
            of the bias
        """
        r_pseudo_inverse = np.linalg.pinv(r_matrix)
        coefficients = r_pseudo_inverse @ y_train
        
        return coefficients

    def _logreg_classification(self) -> Union[LogisticRegression, LogisticRegressionCV]:
        """
        Perform logistic regression training for the classification case

        It trains a logistic regression object to perform classification based
        on the R matrix (activations of the RBFs together with the bias)

        Returns
        -------
        logreg: sklearn.linear_model.LogisticRegression or LogisticRegressionCV
            Scikit-learn logistic regression model already trained
        """
        if self.logisticcv:
            logreg = LogisticRegressionCV(
                Cs=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                penalty='l2' if self.l2 else 'l1',
                solver='saga',
                max_iter=10,
                cv=3,
                random_state=self.random_state,
            )
        else:
            logreg = LogisticRegression(
                penalty='l2' if self.l2 else 'l1',
                C=1 / self.eta,
                solver='saga',
                max_iter=10,
                random_state=self.random_state,
            )

        return logreg

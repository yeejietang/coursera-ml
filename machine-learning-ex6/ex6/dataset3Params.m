function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

% different possible values for C and sigma
step_values = [0.01 0.03 0.1 0.3 1 3 10 30];

% initialize matrix of (C value, sigma value, error value)
error_matrix = zeros(length(step_values)^2, 3);

% initialize error_matrix row index
errmatrix_index = 1;

for c_index = 1:length(step_values)
	for sigma_index = 1:length(step_values)
		% train model using training set
		model_temp = svmTrain(X, y, step_values(c_index), @(x1, x2) gaussianKernel(x1, x2, step_values(sigma_index))); 
		% compute predictions for validation set using trained model
		pred = svmPredict(model_temp, Xval);
		% compute error between predictions and yval
		err = mean(double(pred ~= yval));
		% store C value, sigma value and error value in error matrix
		error_matrix(errmatrix_index, :) = [step_values(c_index) step_values(sigma_index) err];
		% increment for error_matrix row index
		errmatrix_index = errmatrix_index + 1;
	end
end

% find row index of minimum error
[min_err min_err_index] = min(error_matrix(:,3), [], 1);

% assign C to C value with minimum error
C = error_matrix(min_err_index, 1); 
sigma = error_matrix(min_err_index, 2);

% =========================================================================

end

% START: OWN CODE

% Red wine database with wine features and wine quality as target variable.
data = readtable('winequality-red.csv','PreserveVariableNames',true);
data = data{:,:};
data = reshape(data,[],size(data,2));

% Get average of all ratings in list.
col_means = mean(data,1);
rating_mean = col_means(1,size(data,2));

% Set the number of classes we want for our target variable.
num_classes = 2;

% Change target variable to binary- the wine is good (>=rating_mean) or 
% bad (<rating_mean).
if (num_classes==2)
    for i=1:size(data,1)
        if data(i,size(data,2))>=rating_mean
            data(i,size(data,2)) = 1;
        else
            data(i,size(data,2)) = 0;
        end
    end
end

% Set value for number of trees in the random forest.
num_trees = 100;

% Run 10 iterations of the random forest classifier and display mean and 
% std.
accuracies = zeros(10,1);
for i=1:10
    accuracies(i) = random_forest(data,num_classes,num_trees);
end
disp(['Number of classes = ', num2str(num_classes)]);
disp(['Mean = ', num2str(mean(accuracies))]);
disp(['Standard Deviation = ', num2str(std(accuracies))]);

% disp(random_forest(data,num_classes));


% The random forest classifier function.
function accuracy = random_forest(data, num_classes, num_trees)
    rp = randperm(length(data));
    data = data(rp,:);

    % Split data into half training, half testing.
    train_data = data(1:floor(length(data)*2/3),:);
    test_data = data(floor(length(data)*2/3)+1:end,:);

    % Separate training data into feature matrix and target vector.
    train_features = train_data(:,1:size(data,2)-1);
    train_target = train_data(:,size(data,2):end);

    % Create num_trees amount of decision trees with features taken at
    % random using the TreeBagger() function from matlab.
    forest = TreeBagger(num_trees,train_features,train_target);
    % view(forest.Trees{1},'Mode','graph');
    
    % Separate test data into feature matrix and target vector.
    test_features = test_data(:,1:size(data,2)-1);
    test_target = test_data(:,size(data,2):end);

    correct = 0;
    wrong = 0;
    
    % Test accuracy of random forest.
    for test=1:length(test_data)
        counts = zeros(num_classes,1);
        
        % Run the test on every tree in the forest and count number of
        % predictions same as real result and different.
        for tree=1:num_trees
            result = predict(forest.Trees{tree},test_features(test,:));
            result = str2double(result{1});
            counts(result+1) = counts(result+1) + 1;
        end
        
        % If the majority of predictions were correct, increase the correct
        % count, else increase wrong count.
        if (find(counts==max(counts))-1==test_target(test))
            correct = correct + 1;
        else
            wrong = wrong + 1;
        end
    end
    
    % Return accuracy.
    accuracy = correct/(correct+wrong);
end

% END: OWN CODE
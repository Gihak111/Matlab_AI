function model = train_supervised_model(data, numClasses)
    % Check if data has the required fields
    if ~isstruct(data) || ~isfield(data, 'X') || ~isfield(data, 'Y')
        error('Input data should be a structure with fields X and Y.');
    end
    
    % 간단한 지도 학습 모델 학습 (예시로 SVM 사용)
    model = fitcecoc(data.X, data.Y); % 예시로 다중 클래스 SVM 사용
end

% Python 실행 환경 설정
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

% TensorFlow 모듈 로드
import tensorflow.keras.datasets.mnist

% MNIST 데이터셋 로드
[raw_train_images,raw_train_labels,raw_test_images,raw_test_labels] = mnist.load_data();

% 데이터 형식 변환 (MATLAB 호환)
train_images = double(raw_train_images) / 255.0;
test_images = double(raw_test_images) / 255.0;
train_labels = double(raw_train_labels);
test_labels = double(raw_test_labels);

% 데이터 shape 출력
disp('Training images shape:');
disp(size(train_images));
disp('Training labels shape:');
disp(size(train_labels));


% Sequential 모델 생성
model = models.Sequential();

% 첫 번째 Convolutional layer와 MaxPooling layer 추가
model.add(layers.Conv2D(32, [3, 3], activation='relu', input_shape=[28, 28, 1]));
model.add(layers.MaxPooling2D(pool_size=[2, 2]));

% 두 번째 Convolutional layer와 MaxPooling layer 추가
model.add(layers.Conv2D(64, [3, 3], activation='relu'));
model.add(layers.MaxPooling2D(pool_size=[2, 2]));

% Flatten layer 추가
model.add(layers.Flatten());

% Fully connected layer와 출력 layer 추가
model.add(layers.Dense(64, activation='relu'));
model.add(layers.Dense(10, activation='softmax'));

% 모델 요약 출력
model.summary();

% 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']);

% 데이터 차원 맞추기 (TensorFlow는 4차원 형태 [batch_size, height, width, channels]를 요구)
train_images = reshape(train_images, [size(train_images, 1), 28, 28, 1]);
test_images = reshape(test_images, [size(test_images, 1), 28, 28, 1]);

% 모델 학습
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, (test_labels));

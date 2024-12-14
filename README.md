# NeuralNetworkCppfromScratch
### 🤖 Neural Network là gì

Neural Network là một phương thức phổ biến trong lĩnh vực 🌐 trí tuệ nhân tạo, giúp máy tính 🖥️ dự đoán, nhận dạng và xử lý dữ liệu tương tự như 🧠 bộ não con người. Phương thức này còn được gọi là "deep learning" 📚, dựa trên việc kết nối các nơ-ron hoặc các nút với nhau trong một cấu trúc phân lớp 🕸️. Neural Network đã chứng minh khả năng vượt trội trong việc xử lý dữ liệu lớn và phức tạp, được ứng dụng rộng rãi từ phân loại hình ảnh 📷 đến xử lý ngôn ngữ tự nhiên 📖.

Với cơ chế này, hệ thống có thể tự phát hiện và khắc phục các lỗi ⚠️ đã xảy ra thông qua việc tối ưu hóa lặp đi lặp lại 🔄. Neural Network do đó có khả năng giải quyết các vấn đề phức tạp như nhận diện khuôn mặt 😊📸, tóm tắt tài liệu 📄, và thậm chí dự đoán xu hướng thị trường tài chính 📈💸.

---

# 1. 🛠️ Cấu Tạo Của Neural Network

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0cd1a57e-1470-4b50-8a03-54a585466016/268d1e94-a2cb-4a37-8fa7-6a6b05d180ff/image.png)

Mỗi mạng nơ-ron nhân tạo là một perceptron đa tầng và thường bao gồm 3 loại tầng chính:

- **Input Layer (tầng đầu vào):** Tiếp nhận dữ liệu ban đầu 🏗️, chẳng hạn như hình ảnh, âm thanh hoặc văn bản.
- **Output Layer (tầng đầu ra):** Cho kết quả xử lý cuối cùng 📤, ví dụ: phân loại một hình ảnh là mèo 🐱 hoặc chó 🐶.
- **Hidden Layer (tầng ẩn):** Xử lý và suy luận thông qua các liên kết nội giữa các nơ-ron 🔄, thực hiện các tính toán phức tạp để trích xuất đặc trưng từ dữ liệu đầu vào.

Một mạng Neural Network thường có 1 tầng đầu vào và 1 tầng đầu ra, nhưng số lượng tầng ẩn có thể thay đổi tùy thuộc vào độ phức tạp của bài toán 🧩. Số lượng nơ-ron trong từng tầng cũng được thiết kế linh hoạt để tối ưu hóa khả năng xử lý 🎛️.

---

### 2. 📚 Quá Trình Học

Neural Network học thông qua việc huấn luyện dữ liệu 🏋️‍♂️, bao gồm các bước chính như sau:

### 2.1. 🚀 Lan Truyền Tiến (Forward Propagation)

Dữ liệu được chuyển tiến qua các tầng, đến khi tạo ra kết quả đầu ra 📊. Các tham số như **trọng số (weights)** ⚖️ và **bias** ⚙️ được sử dụng để xác định độ ảnh hưởng của từng nơ-ron. Công thức tính toán trong mỗi nơ-ron là:

$$
 z = f(W \cdot x + b) 
$$

Trong đó:

- W : Ma trận trọng số ⚖️
- x: Đầu vào 🎯
- b: Bias ⚙️
- f : Hàm kích hoạt 🔑, ví dụ: ReLU, Sigmoid

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0cd1a57e-1470-4b50-8a03-54a585466016/f0cad791-2d7d-4bdb-a475-c6d2f8337970/image.png)

### 2.2. 🔄 Lan Truyền Ngược (Back Propagation)

Khi kết quả đầu ra không đạt được mong muốn ❌, Neural Network sẽ tính **gradient** của lỗi (loss) để cập nhật lại trọng số theo hướng:

$$
W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}
$$

Trong đó:

- eta: Tốc độ học 🚴‍♀️ (learning rate)
- Cái đạo hàm kia: Gradient của hàm lỗi theo trọng số và bias 🔍

Để hiểu thêm về lan truyền ngược, các bạn có thể tham khảo bài viết này: https://dominhhai.github.io/vi/2018/04/nn-bp/

### 2.3. 🏔️ Gradient Descent

Gradient Descent là phương pháp tối ưu hóa trọng số và bias trong mạng neural bằng cách điều chỉnh các tham số dựa trên gradient của hàm mất mát 📉. Các biến thể phổ biến như **Stochastic Gradient Descent (SGD)** 🌀 và **Adam Optimizer** ⚡ được thiết kế để tăng tốc độ hội tụ và tránh các điểm cực tiểu cục bộ ⛰️, giúp mạng neural học hiệu quả hơn trên dữ liệu huấn luyện.

Để hiểu thêm về lan truyền ngược, các bạn có thể tham khảo bài viết này: https://machinelearningcoban.com/2017/01/12/gradientdescent/

### 2.4. 🧮 Hàm Softmax

Hàm Softmax chủ yếu được sử dụng trong các bài toán phân loại 🗂️ để chuyển kết quả đầu ra thành dạng xác suất. Công thức của hàm là:

$$
 \sigma(z)i = \frac{e^{z_i}}{\sum{j=1}^{N} e^{z_j}} 
$$

Trong đó:

- z_i: Điểm số chưa chuẩn hóa của lớp 🏷️
- N: Tổng số lớp 📦

Hàm Softmax đảm bảo tổng các xác suất bằng 1, hỗ trợ việc giải thích và sử dụng trong các hàm mất mát như Cross-Entropy Loss.

---

### 3. 🎯 Loss Function (Hàm Lỗi)

Loss Function được sử dụng để đánh giá độ chính xác của Neural Network ✅. Các hàm loss phổ biến bao gồm:

- **Mean Squared Error (MSE):**

$$

 L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 
$$

- **Cross-Entropy Loss:**

$$
 L = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i) 
$$

Hàm loss giúp hệ thống nhận biết sự khác biệt giữa đầu ra dự đoán và thực tế, làm cơ sở để cập nhật trọng số 🔄.

---

### 4. 🏗️ Thiết Kế Lớp Ẩn

### 4.1. ⚖️ Weight (Trọng Số) và Bias

Trọng số định độ quan trọng của dữ liệu đầu vào đối với kết quả. Bias giúp điều chỉnh ngưỡng kích hoạt 📏, đảm bảo hệ thống hoạt động hiệu quả ngay cả khi dữ liệu đầu vào bằng 0.

### 4.2. 🔑 Activation Function (Hàm Kích Hoạt)

Trong Neural Network, hàm kích hoạt quyết định cách tạo ra đầu ra từ đầu vào. Các hàm phổ biến bao gồm:

- **ReLU (Rectified Linear Unit):** ⚡ Tăng tốc quá trình huấn luyện bằng cách loại bỏ giá trị âm.
- **Sigmoid:** 📈 Chuyển đổi giá trị đầu vào thành xác suất.
- **Tanh:** 📉 Cân đối đầu ra giữa \(-1\) và \(1\).

### 4.3. 🧮 Phương Trình Lớp Ẩn

Tại tầng ẩn, phương trình tính toán có dạng:

$$
h^{(l)} = f(W^{(l)} \cdot h^{(l-1)} + b^{(l)}) 
$$

Trong đó \( l \) là chỉ số tầng hiện tại 🔢.

---

# Coding

Thư viện Eigen

Để tính toán ma trận một cách hiệu quả, mình sử dụng thư viện Eigen. Chi tiết: https://eigen.tuxfamily.org/index.php?title=Main_Page

Dataset:

Mình sử dụng bộ dataset MNIST - một bộ dữ liệu chuẩn và phổ biến trong machine learning. MNIST chứa các ảnh grayscale của chữ số viết tay từ 0-9. Các bạn có thể tải dataset này từ Kaggle

Kiến trúc dự kiến:

1. Input Layer: 784 nơ-ron (vector hóa ảnh 28×28).
2. Hidden Layer 1: 128 nơ-ron + **ReLU**.
3. Hidden Layer 2: 64 nơ-ron + **ReLU** (tuỳ chọn, nếu muốn mạng sâu hơn).
4. Output Layer: 10 nơ-ron + **Softmax**.

Hàm activations

Ở đây mình sử dụng 2 hàm là ReLU và Softmax, 

Hàm ReLu

- Giải quyết vấn đề **vanishing gradient**, giúp huấn luyện nhanh và ổn định hơn.
- Dễ dàng triển khai và tính toán.

```cpp
    // ReLU Activation Function
    VectorXd relu(const VectorXd &v) {
        return v.array().max(0.0);
    }

    // Derivative of ReLU
    VectorXd relu_derivative(const VectorXd &v) {
        return (v.array() > 0.0).cast<double>();
    }
```

Hàm Softmax

Chuyển các đầu ra thành xác suất, giúp dễ dàng xác định chữ số được dự đoán (chữ số tương ứng với xác suất cao nhất).

```cpp
    // Softmax Activation Function
    VectorXd softmax(const VectorXd &v) {
        VectorXd shifted_v = v.array() - v.maxCoeff();
        VectorXd exp_values = shifted_v.array().exp();
        return exp_values / exp_values.sum();
    }
```

Loss function

Với bài toán phân loại, ta thường sử dụng **Cross-Entropy Loss**:

$$
L = -\sum_{i=1}^{N} y_i \cdot \log(\hat{y}_i)
$$

Trong đó:

- yi: nhãn thật (1-hot encoded)
- ŷi: xác suất dự đoán từ Softmax

Gradient của Softmax + Cross-Entropy có dạng đơn giản:

$$
\frac{\partial L}{\partial z_j} = \hat{y}_j - y_j
$$

(ở đây zj là giá trị đầu ra trước Softmax)

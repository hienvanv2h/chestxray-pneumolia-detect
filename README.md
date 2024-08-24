# Mã nguồn xây dựng API dự đoán bệnh viêm phổi sử dụng mô hình CNN với kiến trúc của ResNet50

### Nguồn dữ liệu ảnh: https://www.kaggle.com/datasets/ahmedhaytham/chest-xray-images-pneumonia-with-new-class

### Mã nguồn để huấn luyện mô hình: Xem trong thư mục **notebooks**

## Giới thiệu về mạng CNN

Mạng nơ-ron tích chập (Convolutional Neural Network), hay còn được gọi là CNN hay ConvNet là một loại mạng nơ-ron sâu đặc biệt được thiết kế để xử lý dữ liệu có cấu trúc dạng lưới, chẳng hạn như hình ảnh. CNN đặc biệt mạnh mẽ trong việc phát hiện các đặc trưng không gian và học các mẫu từ dữ liệu đầu vào. Điều này khiến cho mô hình CNN đóng vai trò quan trọng và được áp dụng trong lĩnh vực nhận diện và phân loại ảnh.
Hình ảnh được lưu trữ trong các thiết bị máy tính là ảnh số. Chúng được thể hiện dưới dạng một ma trận bao gồm vô số các điểm ảnh (pixel) mang các giá trị cường độ sáng khác nhau.

<p align="center" width="100%">
    <img width="33%" src="https://github.com/user-attachments/assets/169a4d7e-095e-4f03-ade5-a7afc13996b8" alt="Biểu diễn ảnh số trên máy tính"> 
</p>

Cấu trúc của một mạng CNN thường bao gồm một chuỗi các lớp chính sau: <br>

- Convolutional Layer (Lớp tích chập): Lớp này áp dụng các bộ lọc (filters hoặc kernels) lên dữ liệu đầu vào để tạo ra các bản đồ đặc trưng (feature maps). Bộ lọc dịch chuyển qua dữ liệu đầu vào và thực hiện phép tính chập, cho phép mạng học các đặc trưng như cạnh, góc, và các đối tượng phức tạp hơn trong dữ liệu.

- Activation Function (Hàm kích hoạt): Sau mỗi lớp tích chập, thường áp dụng hàm kích hoạt phi tuyến tính như ReLU (Rectified Linear Unit) để giới thiệu tính phi tuyến vào mô hình, giúp mạng học được các mối quan hệ phức tạp.

- Pooling Layer (Lớp lấy mẫu): Lớp này giảm kích thước không gian của các bản đồ đặc trưng, giảm số lượng tham số và tính toán trong mạng, và giảm nguy cơ overfitting. Phổ biến nhất là Max Pooling, lấy giá trị lớn nhất trong mỗi vùng cửa sổ nhỏ.

- Fully Connected Layer (Lớp kết nối đầy đủ): Các lớp này thường xuất hiện ở cuối mạng, nơi các bản đồ đặc trưng phẳng được đưa vào một hoặc nhiều lớp kết nối đầy đủ để dự đoán nhãn của dữ liệu đầu vào.

- Output Layer (Lớp đầu ra): Lớp này tạo ra đầu ra cuối cùng của mạng, với số lượng đơn vị (neurons) phù hợp với số lượng lớp nhãn (classes).

<p align="center" width="100%">
    <img width="60%" src="https://github.com/user-attachments/assets/95c8f577-ec5a-4b30-b890-a30b699b2efb" alt="Minh họa các lớp trong mạng CNN"> 
</p>

## Mô tả về tập dữ liệu

Bộ dữ liệu được sắp xếp thành 3 thư mục (Train, Test, Val) và chứa các thư mục con cho từng loại hình ảnh (Bình thường/Viêm phổi do virus/Viêm phổi do vi khuẩn). Có khoảng gần 4500 hình ảnh X-quang (JPEG) đã được cân bằng số lượng giữa các lớp.

<p align="center" width="100%">
    <img width="60%" src="https://github.com/user-attachments/assets/e2ea4898-ccd1-490b-b028-2efd7a75b698" alt="Ảnh minh họa phổi bệnh nhân (bình thường, viêm phổi do vi khuẩn, viêm phổi do virus)"> 
</p>

## Hướng dẫn khởi chạy API

Yêu cầu đã cài đặt **Anaconda** trên máy
<br>

### 1. Thiết lập môi trường chạy

- Xem danh sách môi trường ảo trong conda:

  > conda env list

- Tạo một môi trường ảo:

  > conda create --name chestpneumonia_pred fastapi uvicorn

  Có thể thay opencv-python bằng một thư viện khác đã có trong conda list

- Kích hoạt môi trường ảo:
  > conda activate chestpneumonia_pred

### 2. Cài đặt các thư viện cần thiết cho dự án

- Đầu tiên cần đi đến thư mục chứa code API và file requirements.txt:

  > cd fastapi_pneumonia_detect

- Tải các thư viện trong tệp requirements.txt:

  > pip install -r requirements.txt

### 3. Khởi chạy server trên máy local

- Cách khởi chạy server:

  > uvicorn main:app --reload

- Truy cập đường link sau đó: http://127.0.0.1:8000/docs để thử API bằng Swagger UI.

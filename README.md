# Vietnamese Handwritten Recognition

__Lưu ý__: Để chạy được project, máy chạy khuyến khích có GPU.

## 0. Cấu trúc project

- Trong thư mục `text_recognition`:
    - `data/`: chứa dữ liệu train, public test và data synthesis.
    - `scripts/`: chứa tất cả các scripts dùng để training model khi chạy docker container. Đây cũng là những câu lệnh mà chúng tôi sử dụng để train model trên Kaggle.
    - `saved_models/`: chứa weights của tất cả các model mà chúng tôi đã train.
    - `predictions/`: chứa kết quả dự đoán của tất cả các model mà chúng tôi đã train trên tập public test.

## 1. Set up môi trường

- Sau khi unzip toàn bộ source code, chạy lệnh sau để build docker image:
```bash
docker build -f docker/Dockerfile -t hackathon .
```

- Chạy docker container với gpu:
```bash
docker run  --runtime=nvidia --gpus 1 -it --rm --name pytorch-container --network=host hackathon bash
```

- Chạy docker container không có gpu:
```bash
docker run -it --rm --name pytorch-container --network=host hackathon bash
```

- Sau đó chạy lệnh sau để download dữ liệu từ ban tổ chức cũng như weights của các model mà chúng tôi đã train:
```bash
./prepare_data.sh
```

## 2. Huấn luyện mô hinh
### 2.1. Text recognition

- Kết quả của chúng tôi thu được là từ ensemble của nhiều mô hình khác nhau. Cụ thể là từ các mô hình trong danh sách sau:     ['model5_synth_full', 'model9_full', 'model7_full', 'model4_full', 'model15_full', 'model10_synth_full', 'model10_full', 'model3_full', 'model4_synth_full', 'model2_synth_full', 'model5_full']

- Chạy lệnh sau để _huấn luyện_ và _dự đoán_ kết quả trên tập public test, có thể thay `model5_synth_full` trong câu lệnh bằng tên của mô hình khác trong danh sách trên:
```bash
cd /app/text_recognition
./scripts/model5_synth_full.sh
```





__Chú thích__: Tất cả code đều được chúng tôi viết trên hệ điều hành Windows và chạy mô hình trên Kaggle. Do đó có thể sẽ có một số lỗi khi chạy trên linux. Nếu gặp lỗi, mong các bạn thông cảm và thông báo cho chúng tôi để chúng tôi có thể hỗ trợ và fix lỗi đó.



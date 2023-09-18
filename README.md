# Vietnamese Handwritten Recognition

__Lưu ý__: Để chạy được project, máy chạy khuyến khích có GPU, nếu không có GPU thì cần phải thay đổi một số tham số để chạy docker container.

## 0. Cấu trúc project

- Trong thư mục `text_recognition`:
    - `data/`: chứa dữ liệu train, public test và data synthesis.
    - `scripts/`: chứa tất cả các scripts dùng để training model khi chạy docker container. Đây cũng là những câu lệnh mà chúng tôi sử dụng để train model trên Kaggle.
    - `saved_models/`: chứa weight của tất cả các model mà chúng tôi đã train.
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

- Sau đó chạy lệnh sau để giải nén dữ liệu từ ban tổ chức:
```bash
./prepare_data.sh
```

## 2. Huấn luyện mô hinh

### 2.1. Text recognition

```bash
cd text_recognition
./scripts/model5_synth_full.sh
```


__Chú thích__: Tất cả code đều được chúng tôi viết trên hệ điều hành Windows và chạy mô hình trên Kaggle. Do đó có thể sẽ có một số lỗi khi chạy trên linux. Nếu gặp lỗi, mong các bạn thông cảm và thông báo cho chúng tôi để chúng tôi có thể hỗ trợ và fix lỗi đó.



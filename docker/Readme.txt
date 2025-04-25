・Docker関連の使い方
GPUを積んだ学校の端末と自宅のPCの両方で編集できるようにファイルを構成した
また，データセットのPathを.envファイルで切り替えしなければいけない
使用しない方を#でコメントしておくこと

1. Mac用
docker-compose.yaml
requirements.txt

2. Windows用
docker-compose.gpu.yaml
requirements-gpu.txt

・実行コマンド
Mac用
docker-compose -f docker/docker-compose.yaml up -d

Windows用
docker-compose -f docker/docker-compose.gpu.yaml up -d


※dockerfile関連の変更後は
docker-compose -f docker/docker-compose.yaml up --build -d
docker-compose -f docker/docker-compose.gpu.yaml up --build -d
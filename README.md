### Comit Message Template

fix：バグ修正<br>
hotfix：クリティカルなバグ修正<br>
add：新規（ファイル）機能追加<br>
update：機能修正（バグではない）<br>
change：仕様変更<br>
clean：整理（リファクタリング等）<br>
disable：無効化（コメントアウト等）<br>
remove：削除（ファイル）<br>
upgrade：バージョンアップ<br>
revert：変更取り消し<br>

### pull関連
# ① 今の状態をバックアップブランチに退避（例: backup-2025-05-01）
git checkout -b backup-$(date +"%Y-%m-%d")

# ② 本来の作業ブランチに戻る（例: main）
git checkout main

# ③ リモートの内容に強制リセット（ローカルのmainをorigin/mainに完全一致）
git fetch origin
git reset --hard origin/main

# ④ （必要なら）不要ファイルの掃除
git clean -fd

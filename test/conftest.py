import pytest
import time

@pytest.fixture(scope="module")
def setup_token_generation():
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"最初のトークンが表示されるまでの時間: {elapsed_time}秒")
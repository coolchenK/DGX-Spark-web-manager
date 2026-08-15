from app.security import SecretBox, hash_api_key, mask_secret


def test_secret_box_encrypts_and_decrypts_without_plaintext():
    box = SecretBox("test-secret-key-with-at-least-32-characters")

    encrypted = box.encrypt("provider-secret-token")

    assert "provider-secret-token" not in encrypted
    assert box.decrypt(encrypted) == "provider-secret-token"


def test_api_key_hash_and_mask_are_deterministic():
    assert hash_api_key("dgx_example") == hash_api_key("dgx_example")
    assert hash_api_key("dgx_example") != "dgx_example"
    assert mask_secret("sk-1234567890") == "sk-1...7890"

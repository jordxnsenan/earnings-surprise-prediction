
from earnings_surprise.data.download import calculate_sha256


def test_calculate_sha256(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_bytes(b"abc")

    expected_checksum = (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )

    actual_checksum = calculate_sha256(sample_file)

    assert actual_checksum == expected_checksum

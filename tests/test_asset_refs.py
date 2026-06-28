from sync_workbench.assets.asset_refs import is_probably_absolute_path, normalise_asset_ref


def test_asset_ref_normalisation():
    assert normalise_asset_ref(r"P001\\Session-A\\video.mp4") == "P001/Session-A/video.mp4"
    assert is_probably_absolute_path(r"C:\\data\\x.mp4")
    assert is_probably_absolute_path("/mnt/data/x.mp4")
    assert not is_probably_absolute_path("P001/Session-A/video.mp4")

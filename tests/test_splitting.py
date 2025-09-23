from stellar_platform.data.splitting import object_id_split


def test_object_id_split_stability():
    ids = [f"obj_{i}" for i in range(50)]
    split1 = object_id_split(ids, 0.7, 0.15, 42)
    split2 = object_id_split(ids, 0.7, 0.15, 42)
    assert split1 == split2
    total = set(split1['train']) | set(split1['val']) | set(split1['test'])
    assert total == set(ids)

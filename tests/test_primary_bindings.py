from ordinarylight.wavefront import primary_bindings


def test_primary_layout_variants_preserve_native_abi():
    for native in (False, True):
        for profiling in (False, True):
            bindings = {
                b.binding: b
                for b in primary_bindings(native_textures=native, profiling=profiling)
            }
            assert set(bindings) == set(range(30)) - ({14} if not native else set()) - (
                {15} if not profiling else set()
            )
            assert bindings[0].kind == "acceleration_structure"
            assert {b.binding for b in bindings.values() if b.kind == "image"} == {
                8,
                9,
                19,
                20,
                21,
                22,
            }
            assert bindings[29].count == 16
            if native:
                assert bindings[14].count == 128
            assert bindings[16].access == "read_write"
            assert bindings[17].access == "read"
            assert len({b.name for b in bindings.values()}) == len(bindings)

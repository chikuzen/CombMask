CombMask - Combmask generate filter for Avisynth2.6x


description:
    CombMask is a simple filter that creates a comb mask that can (could) be
    used by other filters like MaskTools2.
    The mask consists of binaries of 0(not combed) and 255(combed).


syntax:
    CombMask(clip, int cthresh, int mthresh, bool sse2)

    cthresh(0 to 255, default is 6):
        spatial combing threshold.

    mthresh(0 to 255, default is 9):
        motion adaptive threshold.


usage:

    LoadPlugin("CombMask.dll)
    LoadPlugin("mt_masktools-26.dll")

    src = SourceFilter("foo\bar\fizz\buzz")
    deint = src.some_deinterlace_filter()
    deint2 = src.another_filter()
    mask = deint.CombMask()
    last = deint.mt_merge(deint2, mask, chroma="process")


reqirement:

    - Avisynth2.6alpha4 or later
    - WindowsXPsp3 / Vista / 7 / 8
    - Microsoft Visual C++ 2010 Redistributable Package
    - SSE2 capable CPU

author:
    Oka Motofumi (chikuzen.mo at gmail dot com)

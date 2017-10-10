#include "CombMask.h"

extern bool has_sse2();
extern bool has_avx2();


static arch_t get_arch(int opt, bool is_avsplus)
{
    if (opt == 0 || !has_sse2()) {
        return NO_SIMD;
    }
#if !defined(__AVX2__)
    return USE_SSE2;
#else
    if (opt == 1 || !has_avx2() || !is_avsplus) {
        return USE_SSE2;
    }
    return USE_AVX2;
#endif
}


static AVSValue __cdecl
create_combmask(AVSValue args, void* user_data, ise_t* env)
{
    enum { CLIP, CTHRESH, MTHRESH, CHROMA, EXPAND, METRIC, OPT };

    PClip clip = args[CLIP].AsClip();
    int metric = args[METRIC].AsInt(0);
    int cth = args[CTHRESH].AsInt(metric == 0 ? 6 : 10);
    int mth = args[MTHRESH].AsInt(9);
    bool ch = args[CHROMA].AsBool(true);
    bool expand = args[EXPAND].AsBool(true);
    bool is_avsplus = env->FunctionExists("SetFilterMTMode");
    arch_t arch = get_arch(args[OPT].AsInt(-1), is_avsplus);

    try{
        return new CombMask(clip, cth, mth, ch, arch, expand, metric, is_avsplus);

    } catch (std::runtime_error& e) {
        env->ThrowError("CombMask: %s", e.what());
    }

    return 0;
}


static AVSValue __cdecl
create_maskedmerge(AVSValue args, void*, IScriptEnvironment* env)
{
    enum { BASE, ALT, MASK, MI, BLOCKX, BLOCKY, CHROMA, OPT };
    try {
        validate(!args[BASE].Defined(), "base clip is not set.");
        validate(!args[ALT].Defined(), "alt clip is not set.");
        validate(!args[MASK].Defined(), "mask clip is not set.");

        PClip base = args[BASE].AsClip();
        PClip alt = args[ALT].AsClip();
        PClip mask = args[MASK].AsClip();

        int mi = args[MI].AsInt(40);
        int bx = args[BLOCKX].AsInt(8);
        int by = args[BLOCKY].AsInt(8);
        bool ch = args[CHROMA].AsBool(true);
        bool is_avsplus = env->FunctionExists("SetFilterMTMode");
        arch_t arch = get_arch(args[OPT].AsInt(-1), is_avsplus);

        return new MaskedMerge(base, alt, mask, mi, bx, by, ch, arch, is_avsplus);
    } catch (std::runtime_error& e) {
        env->ThrowError("MaskedMerge: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl
create_iscombed(AVSValue args, void*, ise_t* env)
{
    enum { CLIP, CTHRESH, MTHRESH, MI, BLOCKX, BLOCKY, METRIC, OPT };
    CombMask* cm = nullptr;

    try {
        AVSValue cf = env->GetVar("current_frame");
        validate(!cf.IsInt(),
                 "This filter can only be used within ConditionalFilter.");
        int n = cf.AsInt();

        PClip clip = args[CLIP].AsClip();
        int metric = args[METRIC].AsInt(0);
        int cth = args[CTHRESH].AsInt(metric == 0 ? 6 : 10);
        int mth = args[MTHRESH].AsInt(9);
        int mi = args[MI].AsInt(80);
        int blockx = args[BLOCKX].AsInt(16);
        int blocky = args[BLOCKY].AsInt(16);
        bool is_avsplus = env->FunctionExists("SetFilterMTMode");
        arch_t arch = get_arch(args[OPT].AsInt(-1), is_avsplus);

        validate(mi < 0 || mi > 128, "MI must be between 0 and 128.");
        validate(blockx != 8 && blockx != 16 && blockx != 32,
                 "blockx must be set to 8, 16 or 32.");
        validate(blocky != 8 && blocky != 16 && blocky != 32,
                 "blocky must be set to 8, 16 or 32.");

        cm = new CombMask(clip, cth, mth, false, arch, false, metric, is_avsplus);

        bool is_combed = (get_check_combed(arch))(
            cm->GetFrame(n, env), mi, blockx, blocky, is_avsplus, env);

        delete cm;

        return AVSValue(is_combed);

    } catch (std::runtime_error& e) {
        if (cm) delete cm;
        env->ThrowError("IsCombed: %s", e.what());
    }
    return 0;
}








const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction(
        "CombMask2", "c[cthresh]i[mthresh]i[chroma]b[expand]b[metric]i[opt]i",
        create_combmask, nullptr);
    env->AddFunction(
        "MaskedMerge2",
        "[base]c[alt]c[mask]c[MI]i[blockx]i[blocky]i[chroma]b[opt]i",
        create_maskedmerge, nullptr);
    env->AddFunction(
        "IsCombed2",
        "c[cthresh]i[mthresh]i[MI]i[blockx]i[blocky]i[metric]i[opt]i",
        create_iscombed, nullptr);

    return "CombMask filter for Avisynth2.6/Avisynth+ version " CMASK_VERSION;
}

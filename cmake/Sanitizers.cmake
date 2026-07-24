function(rlf_enable_sanitizers target_name)
    set(enabled_sanitizers "")

    if(RLF_ENABLE_ASAN)
        list(APPEND enabled_sanitizers "address")
    endif()

    if(RLF_ENABLE_UBSAN)
        list(APPEND enabled_sanitizers "undefined")
    endif()

    if(NOT enabled_sanitizers)
        return()
    endif()

    if(MSVC)
        message(FATAL_ERROR "RLF sanitizers are currently supported with GCC and Clang")
    endif()

    list(JOIN enabled_sanitizers "," sanitizer_list)
    target_compile_options(
        "${target_name}"
        INTERFACE
            "-fsanitize=${sanitizer_list}"
            -fno-omit-frame-pointer
    )
    target_link_options(
        "${target_name}"
        INTERFACE
            "-fsanitize=${sanitizer_list}"
            -fno-omit-frame-pointer
    )
endfunction()

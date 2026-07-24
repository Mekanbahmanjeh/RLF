function(rlf_set_project_warnings target_name)
    if(MSVC)
        target_compile_options(
            "${target_name}"
            INTERFACE
                /W4
                $<$<BOOL:${RLF_WARNINGS_AS_ERRORS}>:/WX>
                /permissive-
        )
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(
            "${target_name}"
            INTERFACE
                -Wall
                -Wextra
                -Wpedantic
                -Wshadow
                -Wconversion
                -Wsign-conversion
                -Wcast-qual
                -Wformat=2
                -Wundef
                -Wdouble-promotion
                -Wnull-dereference
                -Wold-style-cast
                $<$<BOOL:${RLF_WARNINGS_AS_ERRORS}>:-Werror>
        )
    endif()
endfunction()

#!/usr/bin/env bash
# Shell helper: prints a clickable URL to view an image in the browser.
# Source this in your .bashrc/.zshrc, or the server's activate script.
#
# Usage: view /path/to/image.tiff
#        view image.png
#        view .  (lists images in current directory)

VIEW_PORT="${VIEW_PORT:-8089}"

view() {
    if [ $# -eq 0 ]; then
        echo "Usage: view <image_path>"
        echo "       view .  (list images in current dir)"
        return 1
    fi

    local target="$1"

    # List mode
    if [ "$target" = "." ] || [ -d "$target" ]; then
        local dir
        dir="$(cd "$target" && pwd)"
        echo "Images in $dir:"
        find "$dir" -maxdepth 1 \( -iname '*.tif' -o -iname '*.tiff' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.gif' -o -iname '*.webp' \) -printf '%f\n' 2>/dev/null | sort | while read -r f; do
            local full="$dir/$f"
            local encoded
            encoded="$(python3 -c "import urllib.parse; print(urllib.parse.quote('$full'))")"
            local url="http://localhost:${VIEW_PORT}/view?path=${encoded}"
            # OSC 8 hyperlink (clickable in modern terminals)
            printf '\e]8;;%s\e\\  %s\e]8;;\e\\\n' "$url" "$f"
        done
        return 0
    fi

    # Single file mode
    local abspath
    if [[ "$target" = /* ]]; then
        abspath="$target"
    else
        abspath="$(pwd)/$target"
    fi

    if [ ! -f "$abspath" ]; then
        echo "File not found: $abspath"
        return 1
    fi

    local encoded
    encoded="$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" "$abspath")"
    local url="http://localhost:${VIEW_PORT}/view?path=${encoded}"

    # Print clickable link (OSC 8 hyperlink for modern terminals)
    printf '\e]8;;%s\e\\%s\e]8;;\e\\\n' "$url" "$url"
}

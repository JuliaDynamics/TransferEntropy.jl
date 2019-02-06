
## Make sure scipy is installed. Thanks to user sylvaticus in the thread at
# https://discourse.julialang.org/t/pycall-pre-installing-a-python-package-required-by-a-julia-package/3316/15
using PyCall

println("Running build.jl for the TransferEntropy.jl package.")

# Change that to whatever packages you need.
const PACKAGES = ["scipy"]

# Use eventual proxy info
proxy_arg=String[]
if haskey(ENV, "http_proxy")
    push!(proxy_arg, "--proxy")
    push!(proxy_arg, ENV["http_proxy"])
end

# Import pip
try
    @pyimport pip
catch
    # If it is not found, install it
    println("Pip not found on your sytstem. Downloading it.")
    get_pip = joinpath(dirname(@__FILE__), "get-pip.py")
    download("https://bootstrap.pypa.io/get-pip.py", get_pip)
    run(`$(PyCall.python) $(proxy_arg) $get_pip --user`)
end

println("Installing required python packages using pip")
run(`$(PyCall.python) $(proxy_arg) -m pip install --user --upgrade pip setuptools`)
run(`$(PyCall.python) $(proxy_arg) -m pip install --user $(PACKAGES)`)

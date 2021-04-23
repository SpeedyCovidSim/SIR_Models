using Test

push!( LOAD_PATH, "./" )    #Set path to current
using networkFunctions



@testset verbose=true "Network Tests" begin

    @testset "Change State Tests" begin
        networkVertex_dict = Dict(1=>Dict("state"=>"S"))

        network_dict = Dict("S"=>Dict("stateIndex"=>1), "I"=>Dict("stateIndex"=>2))
        isState = convert.(Bool, zeros(2,2))

        changeState!(networkVertex_dict, network_dict, 1, "S", "I", isState)

        @test isState  == [0 1; 0 0]
        @test networkVertex_dict[1]["state"] == "I"
    end
    @testset "Fishface" begin

        @test 1==1
        @test 2==2
    end

end;


1==1;

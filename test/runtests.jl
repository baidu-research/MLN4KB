using Test
using mln4kb
# import mln4kb: ListDict


@testset "ListDict" begin
    a = ListDict{Int64}()
    add!(a, 11)
    add!(a, 101)
    add!(a, 45)
    @test length(a) == 3

    add!(a, 11)
    @test length(a) == 3

    remove!(a, 11)
    @test length(a) == 2

    @test_throws KeyError remove!(a, 11)

    @test length(a) == length(a.items)

    update!(a, 101, 102)
    @test !haskey(a.itemToPosition, 101)
    @test haskey(a.itemToPosition, 102)
    @test 102 in a.items
end
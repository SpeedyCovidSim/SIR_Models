#=
A module containing the function used for conditioning our BPM to data in the August
2021 Outbreak. Note here that it contains an absolute reference to the current
folder location on my machine. The python file it is calling, also uses an
absolute reference to a separate GitHub repository that belongs to Oliver Maclaren
and Frankie Patten-Elliot, containing the conditioning functions.

Author: Joel Trent
=#

module ConditionEnsemble

    using PyCall

    export conditionEnsemble

    function conditionEnsemble(dir, data)

        py"""
        import sys
        sys.path.append("/Users/joeltrent/Documents/GitHub/SIR_Models")
        from conditionEnsemble import conditioningJulia
        """
        conditionEnsemblePy = py"conditioningJulia"

        return conditionEnsemblePy(dir, PyReverseDims(data))
    end

    # function conditionEnsemble(dir)
    #
    #     py"""
    #     import sys
    #     sys.path.append("/Users/joeltrent/Documents/GitHub/SIR_Models")
    #     from conditionEnsemble import conditioningJulia
    #     """
    #     conditionEnsemblePy = py"conditioningJulia"
    #
    #     return conditionEnsemblePy(dir)
    # end
end

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

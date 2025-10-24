"""
Stage configuration and mapping for analysis pipeline
"""
from typing import Dict, List
from app.models.schemas import StageMapping, AnalysisType

class StageConfigService:
    """Service for managing analysis stage configurations"""
    
    def __init__(self):
        self.stage_mappings = self._initialize_stage_mappings()
    
    def _initialize_stage_mappings(self) -> Dict[str, Dict[str, StageMapping]]:
        """Initialize stage mappings for different analysis types"""
        
        # Enhanced analysis stages (new 8-stage structure with parallel AI)
        enhanced_stages = {
            "data_collection_and_analysis": StageMapping(
                stage_name="data_collection_and_analysis",
                display_name="Data Collection and Analysis",
                description="Fetching OHLCV data, fundamentals, and performing enhanced technical and fundamental analysis",
                estimated_duration=25,
                dependencies=[],
                order=1
            ),
            "technical_and_combined_scoring": StageMapping(
                stage_name="technical_and_combined_scoring",
                display_name="Technical and Combined Scoring",
                description="Calculating technical scores and combined scoring",
                estimated_duration=5,
                dependencies=["data_collection_and_analysis"],
                order=2
            ),
            "simple_analysis": StageMapping(
                stage_name="simple_analysis",
                display_name="Simple Analysis",
                description="3-factor analysis: Setup, Catalyst, Confirmation with AI-enhanced risk-reward",
                estimated_duration=5,
                dependencies=["technical_and_combined_scoring"],
                order=3
            ),
            "simple_decision": StageMapping(
                stage_name="simple_decision",
                display_name="Simple Decision",
                description="Clear BUY/WATCH/AVOID decision based on simple analysis",
                estimated_duration=5,
                dependencies=["simple_analysis"],
                order=4
            ),
            "verdict_synthesis": StageMapping(
                stage_name="verdict_synthesis",
                display_name="Verdict Synthesis",
                description="Combining results from simple analysis and decision stages",
                estimated_duration=2,
                dependencies=["simple_analysis", "simple_decision"],
                order=5
            ),
            "final_scoring": StageMapping(
                stage_name="final_scoring",
                display_name="Final Scoring",
                description="Final score calculation and recommendation generation",
                estimated_duration=3,
                dependencies=["verdict_synthesis"],
                order=6
            )
        }
        
        # Basic analysis stages (simplified version)
        basic_stages = {
            "data_collection": StageMapping(
                stage_name="data_collection",
                display_name="Data Collection",
                description="Fetching basic OHLCV data and fundamentals",
                estimated_duration=3,
                dependencies=[],
                order=1
            ),
            "technical_analysis": StageMapping(
                stage_name="technical_analysis",
                display_name="Technical Analysis",
                description="Basic technical indicators calculation",
                estimated_duration=5,
                dependencies=["data_collection"],
                order=2
            ),
            "fundamental_filtering": StageMapping(
                stage_name="fundamental_filtering",
                display_name="Fundamental Filtering",
                description="Basic fundamental sanity checks",
                estimated_duration=2,
                dependencies=["data_collection"],
                order=3
            ),
            "ai_analysis": StageMapping(
                stage_name="ai_analysis",
                display_name="AI Analysis",
                description="Basic AI verdict generation",
                estimated_duration=10,
                dependencies=["technical_analysis", "fundamental_filtering"],
                order=4
            ),
            "final_scoring": StageMapping(
                stage_name="final_scoring",
                display_name="Final Scoring",
                description="Final score calculation",
                estimated_duration=2,
                dependencies=["ai_analysis"],
                order=5
            )
        }
        
        return {
            AnalysisType.ENHANCED.value: enhanced_stages,
            AnalysisType.BASIC.value: basic_stages
        }
    
    def get_stage_mappings(self, analysis_type: AnalysisType) -> Dict[str, StageMapping]:
        """Get stage mappings for a specific analysis type"""
        return self.stage_mappings.get(analysis_type.value, {})
    
    def get_stage_mapping(self, analysis_type: AnalysisType, stage_name: str) -> StageMapping:
        """Get a specific stage mapping"""
        stages = self.get_stage_mappings(analysis_type)
        return stages.get(stage_name)
    
    def get_stage_order(self, analysis_type: AnalysisType) -> List[str]:
        """Get stages in execution order"""
        stages = self.get_stage_mappings(analysis_type)
        return sorted(stages.keys(), key=lambda x: stages[x].order)
    
    def get_total_estimated_time(self, analysis_type: AnalysisType) -> int:
        """Get total estimated time for analysis type"""
        stages = self.get_stage_mappings(analysis_type)
        return sum(stage.estimated_duration for stage in stages.values())
    
    def get_stage_dependencies(self, analysis_type: AnalysisType, stage_name: str) -> List[str]:
        """Get dependencies for a specific stage"""
        stage = self.get_stage_mapping(analysis_type, stage_name)
        return stage.dependencies if stage else []
    
    def can_execute_stage(self, analysis_type: AnalysisType, stage_name: str, completed_stages: List[str]) -> bool:
        """Check if a stage can be executed based on completed dependencies"""
        dependencies = self.get_stage_dependencies(analysis_type, stage_name)
        return all(dep in completed_stages for dep in dependencies)
    
    def get_available_stages(self, analysis_type: AnalysisType, completed_stages: List[str]) -> List[str]:
        """Get stages that can be executed next"""
        all_stages = self.get_stage_order(analysis_type)
        available = []
        
        for stage in all_stages:
            if stage not in completed_stages and self.can_execute_stage(analysis_type, stage, completed_stages):
                available.append(stage)
        
        return available
    
    def get_parallel_stages(self, analysis_type: AnalysisType, completed_stages: List[str]) -> List[str]:
        """Get stages that can be executed in parallel (same order number)"""
        stage_mappings = self.get_stage_mappings(analysis_type)
        parallel_stages = []
        
        # Group stages by order number
        order_groups = {}
        for stage_name, stage_mapping in stage_mappings.items():
            order = stage_mapping.order
            if order not in order_groups:
                order_groups[order] = []
            order_groups[order].append(stage_name)
        
        # Find the next order group that can be executed
        for order in sorted(order_groups.keys()):
            stages_in_order = order_groups[order]
            executable_stages = []
            
            for stage in stages_in_order:
                if stage not in completed_stages and self.can_execute_stage(analysis_type, stage, completed_stages):
                    executable_stages.append(stage)
            
            if executable_stages:
                parallel_stages = executable_stages
                break
        
        return parallel_stages

# Singleton instance
stage_config_service = StageConfigService()

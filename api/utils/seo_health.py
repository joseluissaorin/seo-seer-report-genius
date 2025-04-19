
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class SEOHealthAnalyzer:
    """Analyzer for calculating overall SEO health score."""
    
    def __init__(self):
        self.score_weights = {
            'click_through_rate': 0.15,
            'average_position': 0.20,
            'mobile_friendliness': 0.15,
            'query_diversity': 0.10,
            'content_quality': 0.15,
            'technical_seo': 0.15,
            'backlink_profile': 0.10
        }
    
    def calculate_health_score(self, data_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate the overall SEO health score based on various metrics."""
        scores = {}
        
        # Click-through rate score
        if 'queries' in data_frames:
            queries_df = data_frames['queries']
            ctr_score = self._calculate_ctr_score(queries_df)
            position_score = self._calculate_position_score(queries_df)
            query_diversity_score = self._calculate_query_diversity_score(queries_df)
            scores['click_through_rate'] = ctr_score
            scores['average_position'] = position_score
            scores['query_diversity'] = query_diversity_score
        
        # Mobile friendliness score
        if 'mobile' in data_frames:
            mobile_score = self._calculate_mobile_score(data_frames['mobile'])
            scores['mobile_friendliness'] = mobile_score
        else:
            scores['mobile_friendliness'] = 50  # Default score
        
        # Content quality score (based on meta data if available)
        if 'meta_data' in data_frames:
            content_score = self._calculate_content_quality_score(data_frames['meta_data'])
            scores['content_quality'] = content_score
        else:
            scores['content_quality'] = 50  # Default score
        
        # Technical SEO score (simplified placeholder)
        scores['technical_seo'] = 65  # Default score
        
        # Backlink profile score
        if 'backlinks' in data_frames:
            backlink_score = self._calculate_backlink_score(data_frames['backlinks'])
            scores['backlink_profile'] = backlink_score
        else:
            scores['backlink_profile'] = 50  # Default score
        
        # Calculate weighted average for overall score
        overall_score = 0
        total_weight = 0
        
        for metric, score in scores.items():
            weight = self.score_weights.get(metric, 0)
            overall_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score = overall_score / total_weight
        else:
            overall_score = 50  # Default score
        
        # Generate recommendations based on scores
        recommendations = self._generate_recommendations(scores)
        
        # Create health assessment categories
        health_categories = {
            'excellent': [],
            'good': [],
            'needs_improvement': [],
            'critical': []
        }
        
        for metric, score in scores.items():
            if score >= 80:
                health_categories['excellent'].append(metric)
            elif score >= 60:
                health_categories['good'].append(metric)
            elif score >= 40:
                health_categories['needs_improvement'].append(metric)
            else:
                health_categories['critical'].append(metric)
        
        return {
            'overall_score': round(overall_score, 1),
            'component_scores': scores,
            'recommendations': recommendations,
            'health_categories': health_categories
        }
    
    def _calculate_ctr_score(self, df: pd.DataFrame) -> float:
        """Calculate score based on click-through rate."""
        if df.empty or 'ctr' not in df.columns:
            return 50
        
        # Average CTR
        avg_ctr = df['ctr'].mean()
        
        # Industry average is around 2-3%
        if avg_ctr >= 5:
            return 90
        elif avg_ctr >= 3:
            return 75
        elif avg_ctr >= 2:
            return 60
        elif avg_ctr >= 1:
            return 45
        else:
            return 30
    
    def _calculate_position_score(self, df: pd.DataFrame) -> float:
        """Calculate score based on average position."""
        if df.empty or 'position' not in df.columns:
            return 50
        
        # Average position
        avg_position = df['position'].mean()
        
        # Position score (lower is better)
        if avg_position <= 3:
            return 90
        elif avg_position <= 5:
            return 80
        elif avg_position <= 10:
            return 70
        elif avg_position <= 20:
            return 50
        elif avg_position <= 50:
            return 30
        else:
            return 15
    
    def _calculate_query_diversity_score(self, df: pd.DataFrame) -> float:
        """Calculate score based on query diversity."""
        if df.empty or 'query' not in df.columns:
            return 50
        
        # Number of unique queries
        unique_queries = df['query'].nunique()
        
        # Query diversity score
        if unique_queries >= 1000:
            return 90
        elif unique_queries >= 500:
            return 80
        elif unique_queries >= 200:
            return 70
        elif unique_queries >= 100:
            return 60
        elif unique_queries >= 50:
            return 50
        else:
            return 40
    
    def _calculate_mobile_score(self, df: pd.DataFrame) -> float:
        """Calculate score based on mobile friendliness."""
        if df.empty:
            return 50
        
        # Check if we have direct mobile scores
        if 'mobile_friendly_score' in df.columns:
            avg_score = df['mobile_friendly_score'].mean()
            return avg_score
        
        # Check if we have mobile usability issues
        if 'mobile_usability' in df.columns:
            # Count pages with mobile issues
            pages_with_issues = df[df['mobile_usability'] == 'issues'].shape[0]
            total_pages = df.shape[0]
            
            if total_pages == 0:
                return 50
            
            # Calculate percentage of pages with issues
            issue_percentage = (pages_with_issues / total_pages) * 100
            
            # Mobile score
            if issue_percentage <= 5:
                return 90
            elif issue_percentage <= 15:
                return 75
            elif issue_percentage <= 30:
                return 60
            elif issue_percentage <= 50:
                return 40
            else:
                return 25
        
        return 50  # Default score
    
    def _calculate_content_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate score based on content quality metrics."""
        if df.empty:
            return 50
        
        score = 50  # Default score
        factors = 0
        
        # Meta description quality
        if 'meta_description' in df.columns:
            # Count non-empty meta descriptions
            meta_counts = df['meta_description'].notna().sum()
            meta_percentage = (meta_counts / df.shape[0]) * 100
            
            meta_score = 0
            if meta_percentage >= 90:
                meta_score = 90
            elif meta_percentage >= 75:
                meta_score = 75
            elif meta_percentage >= 50:
                meta_score = 60
            elif meta_percentage >= 25:
                meta_score = 40
            else:
                meta_score = 25
            
            score += meta_score
            factors += 1
            
            # Meta description length (ideal is 120-158 characters)
            df['meta_length'] = df['meta_description'].fillna('').apply(len)
            optimal_meta = ((df['meta_length'] >= 120) & (df['meta_length'] <= 158)).mean() * 100
            
            meta_length_score = 0
            if optimal_meta >= 80:
                meta_length_score = 90
            elif optimal_meta >= 60:
                meta_length_score = 75
            elif optimal_meta >= 40:
                meta_length_score = 60
            elif optimal_meta >= 20:
                meta_length_score = 40
            else:
                meta_length_score = 25
            
            score += meta_length_score
            factors += 1
        
        # Title quality
        if 'title' in df.columns:
            # Count non-empty titles
            title_counts = df['title'].notna().sum()
            title_percentage = (title_counts / df.shape[0]) * 100
            
            title_score = 0
            if title_percentage >= 95:
                title_score = 90
            elif title_percentage >= 80:
                title_score = 75
            elif title_percentage >= 60:
                title_score = 60
            elif title_percentage >= 40:
                title_score = 40
            else:
                title_score = 25
            
            score += title_score
            factors += 1
            
            # Title length (ideal is 50-60 characters)
            df['title_length'] = df['title'].fillna('').apply(len)
            optimal_title = ((df['title_length'] >= 50) & (df['title_length'] <= 60)).mean() * 100
            
            title_length_score = 0
            if optimal_title >= 80:
                title_length_score = 90
            elif optimal_title >= 60:
                title_length_score = 75
            elif optimal_title >= 40:
                title_length_score = 60
            elif optimal_title >= 20:
                title_length_score = 40
            else:
                title_length_score = 25
            
            score += title_length_score
            factors += 1
        
        # Calculate average score
        if factors > 0:
            return score / factors
        else:
            return 50
    
    def _calculate_backlink_score(self, df: pd.DataFrame) -> float:
        """Calculate score based on backlink profile."""
        if df.empty:
            return 50
        
        score = 50  # Default score
        factors = 0
        
        # Backlink quantity
        if 'backlinks' in df.columns:
            total_backlinks = df['backlinks'].sum()
            
            backlink_score = 0
            if total_backlinks >= 10000:
                backlink_score = 90
            elif total_backlinks >= 5000:
                backlink_score = 80
            elif total_backlinks >= 1000:
                backlink_score = 70
            elif total_backlinks >= 500:
                backlink_score = 60
            elif total_backlinks >= 100:
                backlink_score = 50
            elif total_backlinks >= 50:
                backlink_score = 40
            else:
                backlink_score = 30
            
            score += backlink_score
            factors += 1
        
        # Domain authority
        if 'domain_authority' in df.columns:
            avg_da = df['domain_authority'].mean()
            
            da_score = 0
            if avg_da >= 70:
                da_score = 90
            elif avg_da >= 50:
                da_score = 80
            elif avg_da >= 40:
                da_score = 70
            elif avg_da >= 30:
                da_score = 60
            elif avg_da >= 20:
                da_score = 50
            elif avg_da >= 10:
                da_score = 40
            else:
                da_score = 30
            
            score += da_score
            factors += 1
        
        # Calculate average score
        if factors > 0:
            return score / factors
        else:
            return 50
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate recommendations based on scores."""
        recommendations = []
        
        # CTR recommendations
        if 'click_through_rate' in scores:
            ctr_score = scores['click_through_rate']
            if ctr_score < 60:
                recommendations.append({
                    'category': 'click_through_rate',
                    'title': 'Improve Click-Through Rate',
                    'description': 'Enhance meta titles and descriptions to make them more compelling and relevant to search queries.',
                    'priority': 'high' if ctr_score < 40 else 'medium'
                })
        
        # Position recommendations
        if 'average_position' in scores:
            position_score = scores['average_position']
            if position_score < 70:
                recommendations.append({
                    'category': 'average_position',
                    'title': 'Improve Search Rankings',
                    'description': 'Focus on improving content quality, enhancing keyword targeting, and building quality backlinks to improve rankings.',
                    'priority': 'high' if position_score < 50 else 'medium'
                })
        
        # Mobile recommendations
        if 'mobile_friendliness' in scores:
            mobile_score = scores['mobile_friendliness']
            if mobile_score < 70:
                recommendations.append({
                    'category': 'mobile_friendliness',
                    'title': 'Enhance Mobile Experience',
                    'description': 'Address mobile usability issues by improving responsive design, optimizing page speed, and ensuring tap targets are properly sized.',
                    'priority': 'high' if mobile_score < 50 else 'medium'
                })
        
        # Content quality recommendations
        if 'content_quality' in scores:
            content_score = scores['content_quality']
            if content_score < 70:
                recommendations.append({
                    'category': 'content_quality',
                    'title': 'Improve Content Quality',
                    'description': 'Optimize meta descriptions and titles for appropriate length and relevance to improve CTR and user experience.',
                    'priority': 'high' if content_score < 50 else 'medium'
                })
        
        # Backlink recommendations
        if 'backlink_profile' in scores:
            backlink_score = scores['backlink_profile']
            if backlink_score < 60:
                recommendations.append({
                    'category': 'backlink_profile',
                    'title': 'Strengthen Backlink Profile',
                    'description': 'Develop a strategy to acquire more high-quality backlinks from authoritative domains relevant to your industry.',
                    'priority': 'high' if backlink_score < 40 else 'medium'
                })
        
        return recommendations

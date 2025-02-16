import os
from dotenv import load_dotenv
import anthropic
from typing import Dict, List
import time
import json
from collections import Counter

# Load environment variables
load_dotenv()

class CheckoutAnalyzer:
    def __init__(self):
        self.client = anthropic.Client(api_key=os.getenv('CLAUDE_API_KEY'))
        self.friction_thresholds = {
            'add_to_cart': 1.0,      # seconds
            'initiate_checkout': 1.5,
            'customer_info': 1.0,
            'shipping_info': 1.0
        }
        
        self.industry_benchmarks = {
            'avg_completion_time': 2.1,  # minutes
            'cart_abandonment_rate': 0.69,
            'mobile_conversion_rate': 0.15,
            'desktop_conversion_rate': 0.23,
            'optimal_steps': 4,
            'payment_success_rate': 0.95,
            'form_completion_rate': 0.82
        }

        self.checkout_patterns = {
            'guest_checkout': {'frequency': 0.65, 'avg_steps': 5},
            'returning_user': {'frequency': 0.35, 'avg_steps': 3},
            'mobile_user': {'frequency': 0.72, 'avg_steps': 6},
            'desktop_user': {'frequency': 0.28, 'avg_steps': 4}
        }
        
        # Expanded user scenarios
        self.user_scenarios = {
            'new_customer': {
                'frequency': 0.45,
                'characteristics': {
                    'hesitation_rate': 0.3,
                    'form_error_rate': 0.25,
                    'cart_abandonment_risk': 0.75,
                    'price_sensitivity': 'high',
                    'avg_completion_time': 4.5  # minutes
                }
            },
            'returning_customer': {
                'frequency': 0.30,
                'characteristics': {
                    'hesitation_rate': 0.1,
                    'form_error_rate': 0.08,
                    'cart_abandonment_risk': 0.35,
                    'price_sensitivity': 'medium',
                    'avg_completion_time': 2.0
                }
            },
            'mobile_user_rush': {
                'frequency': 0.15,
                'characteristics': {
                    'hesitation_rate': 0.4,
                    'form_error_rate': 0.30,
                    'cart_abandonment_risk': 0.80,
                    'price_sensitivity': 'medium',
                    'avg_completion_time': 3.5,
                    'network_stability': 'variable'
                }
            },
            'international_customer': {
                'frequency': 0.05,
                'characteristics': {
                    'hesitation_rate': 0.35,
                    'form_error_rate': 0.28,
                    'cart_abandonment_risk': 0.70,
                    'price_sensitivity': 'high',
                    'avg_completion_time': 5.0,
                    'language_barriers': True,
                    'payment_method_constraints': True
                }
            },
            'bulk_business_buyer': {
                'frequency': 0.03,
                'characteristics': {
                    'hesitation_rate': 0.15,
                    'form_error_rate': 0.12,
                    'cart_abandonment_risk': 0.25,
                    'price_sensitivity': 'low',
                    'avg_completion_time': 6.0,
                    'requires_invoice': True,
                    'multiple_shipping_addresses': True
                }
            },
            'senior_user': {
                'frequency': 0.02,
                'characteristics': {
                    'hesitation_rate': 0.45,
                    'form_error_rate': 0.35,
                    'cart_abandonment_risk': 0.65,
                    'price_sensitivity': 'medium',
                    'avg_completion_time': 7.0,
                    'accessibility_needs': True
                }
            }
        }

        # Add specific friction points for each scenario
        self.scenario_friction_points = {
            'new_customer': [
                'account_creation_hesitation',
                'form_field_confusion',
                'shipping_cost_surprise',
                'payment_method_trust'
            ],
            'returning_customer': [
                'password_reset_needed',
                'saved_card_expired',
                'address_update_required'
            ],
            'mobile_user_rush': [
                'network_timeout',
                'form_field_misclick',
                'session_expiry',
                'payment_verification_delay'
            ],
            'international_customer': [
                'currency_conversion_confusion',
                'address_format_mismatch',
                'shipping_restriction_discovery',
                'payment_method_unavailability'
            ],
            'bulk_business_buyer': [
                'quantity_limit_restrictions',
                'bulk_pricing_calculation',
                'multiple_shipping_setup',
                'invoice_requirement_handling'
            ],
            'senior_user': [
                'text_size_readability',
                'complex_navigation',
                'timeout_frequency',
                'payment_method_complexity'
            ]
        }

    def analyze_checkout(self, checkout_data: Dict) -> Dict:
        """Analyze the checkout experience and generate a report"""
        grade = 'A'
        friction_points = []
        recommendations = []
        
        # Analyze overall success
        if not checkout_data['success']:
            grade = 'F'
            friction_points.append("Checkout failed to complete")
            recommendations.append("Investigate failed checkout steps")
        
        # Analyze timing of each step
        for step in checkout_data['steps']:
            if step['time'] > self.friction_thresholds[step['name']]:
                grade = 'B' if grade == 'A' else grade
                friction_points.append(f"{step['name']} took longer than expected ({step['time']:.2f}s)")
                recommendations.append(f"Optimize {step['name']} step for better performance")
        
        # Analyze total checkout time
        if checkout_data['total_time'] > 4.0:  # More than 4 seconds total
            grade = 'C' if grade in ['A', 'B'] else grade
            friction_points.append("Total checkout time is too long")
            recommendations.append("Review overall checkout flow for optimization opportunities")
        
        return {
            'grade': grade,
            'friction_points': friction_points,
            'recommendations': recommendations,
            'metrics': {
                'total_time': checkout_data['total_time'],
                'step_times': {step['name']: step['time'] for step in checkout_data['steps']},
                'success_rate': 1.0 if checkout_data['success'] else 0.0
            }
        }
    
    def generate_report(self, training_data: List[Dict]) -> str:
        """Generate one comprehensive final report with actionable insights"""
        
        # Gather all metrics
        metrics = {
            'total_episodes': len(training_data),
            'final_success_rate': sum(1 for ep in training_data[-100:] if ep['success']) / 100,
            'optimal_path': self._find_optimal_path(training_data),
            'average_steps': sum(ep['steps'] for ep in training_data[-100:]) / 100,
            'friction_points': self._identify_friction_points(training_data),
            'error_patterns': self._get_common_errors(training_data),
            'learning_progression': self._analyze_learning_progression(training_data),
            'benchmark_comparison': self._compare_to_benchmarks(training_data),
            'user_patterns': self._analyze_user_patterns(training_data),
            'conversion_impact': self._estimate_conversion_impact(training_data),
            'revenue_projection': self._project_revenue_impact(training_data)
        }

        prompt = f"""
        As an expert in e-commerce optimization and machine learning, provide a comprehensive analysis comparing this checkout flow simulation with industry standards.

        Performance vs Industry Benchmarks:
        {metrics['benchmark_comparison']}

        User Pattern Analysis:
        {metrics['user_patterns']}

        Projected Impact:
        - Conversion Impact: {metrics['conversion_impact']}
        - Revenue Impact: {metrics['revenue_projection']}

        Training Results Summary:
        - Total Episodes: {metrics['total_episodes']}
        - Final Success Rate: {metrics['final_success_rate']*100:.1f}%
        - Average Steps to Complete: {metrics['average_steps']:.1f}
        - Optimal Action Sequence: {metrics['optimal_path']}
        - Learning Progression: {metrics['learning_progression']}

        Please provide a detailed, actionable report covering:

        1. Executive Summary
        - Key findings from the simulation
        - Most significant opportunities for improvement
        - Expected impact of recommended changes

        2. Checkout Flow Analysis
        - Evaluation of current checkout sequence
        - Comparison with e-commerce best practices
        - Identified friction points and their impact
        - Potential bottlenecks and their solutions

        3. Specific Recommendations
        - Prioritized list of improvements
        - Technical implementation requirements
        - Expected impact on conversion rates
        - Risk assessment for each change

        4. Implementation Roadmap
        - Immediate actions (next 30 days)
        - Medium-term improvements (60-90 days)
        - Long-term optimization strategy
        - Success metrics to track

        Additional Analysis Requirements:
        1. Compare performance against industry leaders (Amazon, Shopify, etc.)
        2. Identify competitive advantages and disadvantages
        3. Suggest innovative checkout optimizations
        4. Provide mobile-specific recommendations
        5. Address regional/international considerations
        6. Analyze potential for emerging technologies (one-click, digital wallets, etc.)
        7. Consider accessibility and inclusive design impacts
        """

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content

    def analyze_episode(self, episode_data: Dict) -> str:
        """Analyze a single training episode"""
        prompt = f"""
        Analyze this checkout episode data and provide detailed insights:

        Episode Number: {episode_data['episode']}
        Total Reward: {episode_data['total_reward']}
        Steps Taken: {episode_data['steps']}
        Actions: {episode_data['actions']}
        Errors: {episode_data['errors']}
        Completion Time: {episode_data['completion_time']}s

        Please provide:
        1. Identification of specific friction points
        2. Analysis of why these points caused issues
        3. Detailed recommendations for improvement
        4. Comparison to optimal checkout flow
        """

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content

    def analyze_training_session(self, training_data: List[Dict]) -> str:
        """Analyze entire training session for patterns and insights"""
        session_summary = {
            'total_episodes': len(training_data),
            'success_rate': sum(1 for ep in training_data if ep['success']) / len(training_data),
            'avg_steps': sum(ep['steps'] for ep in training_data) / len(training_data),
            'common_errors': self._get_common_errors(training_data),
            'performance_trend': self._analyze_performance_trend(training_data)
        }

        prompt = f"""
        Analyze this checkout training session data and provide comprehensive insights:

        Session Summary:
        {json.dumps(session_summary, indent=2)}

        Please provide:
        1. Overall assessment of the checkout flow
        2. Pattern analysis of common friction points
        3. Systematic recommendations for improvement
        4. Prioritized list of optimizations
        5. Impact analysis of potential improvements
        """

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content

    def _get_common_errors(self, training_data: List[Dict]) -> Dict:
        """Analyze common error patterns"""
        error_counts = {}
        for episode in training_data:
            for error in episode.get('errors', []):
                error_counts[error] = error_counts.get(error, 0) + 1
        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True))

    def _analyze_performance_trend(self, training_data: List[Dict]) -> Dict:
        """Analyze performance trends over time"""
        return {
            'early_success_rate': self._get_success_rate(training_data[:100]),
            'late_success_rate': self._get_success_rate(training_data[-100:]),
            'improvement': self._get_success_rate(training_data[-100:]) - self._get_success_rate(training_data[:100])
        }

    def _get_success_rate(self, episodes: List[Dict]) -> float:
        """Calculate success rate for a set of episodes"""
        return sum(1 for ep in episodes if ep['success']) / len(episodes)

    def analyze_batch(self, batch_data: List[Dict]) -> Dict:
        """Analyze a batch of episodes with detailed insights"""
        batch_summary = {
            'episodes': len(batch_data),
            'success_rate': sum(1 for ep in batch_data if ep['success']) / len(batch_data),
            'avg_reward': sum(ep['total_reward'] for ep in batch_data) / len(batch_data),
            'avg_steps': sum(ep['steps'] for ep in batch_data) / len(batch_data),
            'errors': [error for ep in batch_data for error in ep['errors']],
            'action_patterns': self._analyze_action_patterns(batch_data),
            'learning_curve': self._calculate_learning_curve(batch_data)
        }

        prompt = f"""
        You are an expert in e-commerce checkout optimization and machine learning. Analyze this batch of {batch_summary['episodes']} checkout episodes and provide detailed insights.

        Performance Metrics:
        - Success Rate: {batch_summary['success_rate']*100:.1f}%
        - Average Reward: {batch_summary['avg_reward']:.1f}
        - Average Steps: {batch_summary['avg_steps']:.1f}
        - Total Errors: {len(batch_summary['errors'])}
        
        Action Patterns:
        {batch_summary['action_patterns']}
        
        Learning Curve:
        {batch_summary['learning_curve']}

        Please provide a detailed analysis in the following format:

        1. Learning Progress:
        - Analyze how well the agent is learning
        - Identify any plateaus or breakthroughs
        - Compare performance to expected benchmarks

        2. Checkout Flow Analysis:
        - Evaluate the efficiency of the checkout sequence
        - Identify any unnecessary steps or redundancies
        - Compare to industry best practices

        3. Friction Points:
        - List specific points where users might experience friction
        - Analyze the impact of each friction point
        - Prioritize which issues need immediate attention

        4. Technical Insights:
        - Evaluate the agent's learning strategy
        - Identify any potential overfitting or underfitting
        - Suggest improvements to the learning process

        5. Specific Recommendations:
        - Provide actionable steps to improve the checkout flow
        - Suggest technical improvements to the learning process
        - Recommend A/B tests or experiments to validate changes

        6. Risk Assessment:
        - Identify potential failure modes
        - Evaluate security implications
        - Suggest risk mitigation strategies

        Please be specific and provide concrete examples where possible.
        """

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Get the content from the response
        analysis = response.content[0].text if isinstance(response.content, list) else response.content
        
        print("\n=== Detailed Analysis ===\n")
        print(analysis)
        
        # Extract and format specific recommendations
        recommendations = []
        for line in analysis.split('\n'):
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'could', 'improve']):
                if line.strip() and not line.strip().endswith(':'):
                    recommendations.append(line.strip())
        
        print("\n=== Key Recommendations ===\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        return {
            'insights': analysis,
            'metrics': batch_summary,
            'recommendations': recommendations
        }

    def _analyze_action_patterns(self, batch_data: List[Dict]) -> str:
        """Analyze patterns in action sequences"""
        action_sequences = [ep['actions'] for ep in batch_data]
        common_patterns = {}
        for seq in action_sequences:
            pattern = tuple(seq)
            common_patterns[pattern] = common_patterns.get(pattern, 0) + 1
        
        return "\n".join([f"Pattern {pattern}: {count} times" 
                         for pattern, count in sorted(common_patterns.items(), 
                         key=lambda x: x[1], reverse=True)[:5]])

    def _calculate_learning_curve(self, batch_data: List[Dict]) -> str:
        """Calculate and format learning curve data"""
        rewards = [ep['total_reward'] for ep in batch_data]
        window_size = 10
        moving_avg = [sum(rewards[i:i+window_size])/window_size 
                     for i in range(0, len(rewards)-window_size+1)]
        return f"Moving average (window={window_size}): {moving_avg}"

    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from Claude's analysis"""
        recommendations = []
        for line in analysis.split('\n'):
            if 'recommend' in line.lower() or 'suggest' in line.lower():
                recommendations.append(line.strip())
        return recommendations

    def _find_optimal_path(self, training_data: List[Dict]) -> List[int]:
        """Find the most successful action sequence"""
        successful_episodes = [ep for ep in training_data if ep['total_reward'] >= 230]
        if not successful_episodes:
            return []
        
        # Get the shortest successful sequence
        shortest = min(successful_episodes, key=lambda x: len(x['actions']))
        return shortest['actions']

    def _identify_friction_points(self, training_data: List[Dict]) -> Dict:
        """Identify common points where the agent struggled"""
        friction_points = {}
        for episode in training_data:
            if episode['total_reward'] < 200:
                for i, action in enumerate(episode['actions']):
                    if i > 0:
                        transition = f"{episode['actions'][i-1]}_to_{action}"
                        friction_points[transition] = friction_points.get(transition, 0) + 1
        return friction_points

    def _analyze_learning_progression(self, training_data: List[Dict]) -> str:
        """Analyze how the agent's performance improved over time"""
        quarter_size = len(training_data) // 4
        quarters = [training_data[i:i + quarter_size] 
                   for i in range(0, len(training_data), quarter_size)]
        
        progression = []
        for i, quarter in enumerate(quarters):
            avg_reward = sum(ep['total_reward'] for ep in quarter) / len(quarter)
            avg_steps = sum(ep['steps'] for ep in quarter) / len(quarter)
            progression.append(f"Q{i+1}: {avg_reward:.1f} reward, {avg_steps:.1f} steps")
        
        return " | ".join(progression)

    def _compare_to_benchmarks(self, training_data: List[Dict]) -> Dict:
        """Compare performance against industry benchmarks"""
        avg_steps = sum(ep['steps'] for ep in training_data[-100:]) / 100
        completion_time = avg_steps * 0.1  # Assuming 100ms per step
        
        return {
            'steps_vs_optimal': f"{avg_steps:.1f} vs {self.industry_benchmarks['optimal_steps']} (industry)",
            'completion_time': f"{completion_time:.1f}m vs {self.industry_benchmarks['avg_completion_time']}m (industry)",
            'success_rate_vs_industry': f"{self._calculate_success_rate(training_data):.1%} vs {self.industry_benchmarks['payment_success_rate']:.1%}"
        }

    def _analyze_user_patterns(self, training_data: List[Dict]) -> Dict:
        """Analyze patterns against typical user behaviors"""
        return {
            'pattern_alignment': self._check_pattern_alignment(training_data),
            'mobile_optimization': self._assess_mobile_readiness(training_data),
            'user_type_performance': self._analyze_by_user_type(training_data)
        }

    def _check_pattern_alignment(self, training_data: List[Dict]) -> Dict:
        """Check how well the agent's behavior aligns with expected patterns"""
        successful_episodes = [ep for ep in training_data[-100:] if ep['total_reward'] >= 220]
        if not successful_episodes:
            return {'alignment': 0.0, 'patterns': []}
        
        # Analyze common action sequences
        action_sequences = [tuple(ep['actions']) for ep in successful_episodes]
        common_sequences = Counter(action_sequences).most_common(3)
        
        return {
            'alignment': len(successful_episodes) / 100,
            'patterns': [
                {
                    'sequence': list(seq),
                    'frequency': count / len(successful_episodes)
                }
                for seq, count in common_sequences
            ]
        }

    def _assess_mobile_readiness(self, training_data: List[Dict]) -> Dict:
        """Assess the checkout flow's mobile optimization"""
        avg_steps = sum(ep['steps'] for ep in training_data[-100:]) / 100
        
        return {
            'step_efficiency': 'Good' if avg_steps <= 5 else 'Needs Improvement',
            'estimated_mobile_completion_time': f"{avg_steps * 0.15:.1f}m",  # 50% longer on mobile
            'potential_mobile_issues': [
                'Form field size optimization needed' if avg_steps > 5 else None,
                'Touch target spacing review recommended' if avg_steps > 4 else None,
                'Mobile payment integration suggested' if avg_steps > 3 else None
            ]
        }

    def _analyze_by_user_type(self, training_data: List[Dict]) -> Dict:
        """Analyze performance for different user types"""
        return {
            user_type: {
                'avg_steps': self._calculate_avg_completion_time(training_data, user_type),
                'success_rate': self._calculate_completion_rate(training_data, user_type),
                'friction_points': self._identify_scenario_friction_points(training_data, user_type)
            }
            for user_type in self.user_scenarios.keys()
        }

    def _estimate_conversion_impact(self, training_data: List[Dict]) -> float:
        """Estimate potential conversion rate improvements"""
        current_performance = self._calculate_success_rate(training_data)
        industry_avg = self.industry_benchmarks['payment_success_rate']
        
        potential_improvement = (industry_avg - current_performance) / current_performance
        return potential_improvement

    def _project_revenue_impact(self, training_data: List[Dict]) -> str:
        """Project potential revenue impact of improvements"""
        # Assuming average order value and monthly transaction volume
        avg_order_value = 85  # dollars
        monthly_transactions = 10000
        conversion_improvement = self._estimate_conversion_impact(training_data)
        
        monthly_revenue_impact = avg_order_value * monthly_transactions * conversion_improvement
        return f"Projected monthly revenue impact: ${monthly_revenue_impact:,.2f}"

    def _analyze_scenario_specific_patterns(self, training_data: List[Dict]) -> Dict:
        """Analyze patterns for each user scenario"""
        scenario_analysis = {}
        
        for scenario, details in self.user_scenarios.items():
            # Calculate scenario-specific metrics
            completion_rate = self._calculate_completion_rate(training_data, scenario)
            avg_time = self._calculate_avg_completion_time(training_data, scenario)
            friction_points = self._identify_scenario_friction_points(training_data, scenario)
            
            scenario_analysis[scenario] = {
                'completion_rate': completion_rate,
                'avg_completion_time': avg_time,
                'common_friction_points': friction_points,
                'improvement_opportunities': self._generate_scenario_recommendations(scenario, friction_points)
            }
        
        return scenario_analysis

    def _generate_scenario_recommendations(self, scenario: str, friction_points: List[str]) -> List[str]:
        """Generate specific recommendations for each user scenario"""
        recommendations = []
        
        if scenario == 'new_customer':
            recommendations.extend([
                'Implement guest checkout with account creation option at end',
                'Add form field tooltips and validation hints',
                'Show shipping cost calculator early in process',
                'Display security badges and payment guarantees'
            ])
        elif scenario == 'mobile_user_rush':
            recommendations.extend([
                'Implement form auto-fill and smart defaults',
                'Add progress save feature for network issues',
                'Extend session timeouts for mobile users',
                'Optimize form field size for touch input'
            ])
        # ... add recommendations for other scenarios ...
        
        return recommendations

    def _calculate_success_rate(self, training_data: List[Dict]) -> float:
        """Calculate the success rate from training data"""
        if not training_data:
            return 0.0
        # Consider last 100 episodes for current performance
        recent_data = training_data[-100:]
        successful_episodes = sum(1 for ep in recent_data if ep['total_reward'] >= 220)
        return successful_episodes / len(recent_data)

    def _calculate_completion_rate(self, training_data: List[Dict], scenario: str) -> float:
        """Calculate completion rate for specific scenario"""
        if not training_data:
            return 0.0
        recent_data = training_data[-100:]
        # For now, treat all data as applicable to the scenario
        successful_episodes = sum(1 for ep in recent_data if ep['total_reward'] >= 220)
        return successful_episodes / len(recent_data)

    def _calculate_avg_completion_time(self, training_data: List[Dict], scenario: str) -> float:
        """Calculate average completion time for specific scenario"""
        if not training_data:
            return 0.0
        recent_data = training_data[-100:]
        # Assume each step takes 0.1 minutes
        total_time = sum(ep['steps'] * 0.1 for ep in recent_data)
        return total_time / len(recent_data)

    def _identify_scenario_friction_points(self, training_data: List[Dict], scenario: str) -> List[str]:
        """Identify friction points for specific scenario"""
        # Return predefined friction points for the scenario
        return self.scenario_friction_points.get(scenario, [])

class CheckoutEnvironment:
    def __init__(self):
        self.state.update({
            'payment_method_selected': False,
            'promo_code_applied': False,
            'address_validated': False,
            'inventory_checked': False,
            'mobile_device': False,  # Simulate different devices
            'network_latency': 0,    # Simulate network conditions
            'user_type': 'new'       # new/returning user scenarios
        })

class ABTestingAnalyzer:
    def __init__(self):
        self.variants = {
            'original': CheckoutEnvironment(),
            'simplified': SimplifiedCheckoutEnvironment(),
            'one_page': OnePageCheckoutEnvironment()
        }
    
    def compare_variants(self, episodes=1000):
        results = {}
        for name, env in self.variants.items():
            agent = CheckoutAgent(env)
            training_data = agent.train(episodes)
            results[name] = self.analyze_performance(training_data)
        return results

class UserBehaviorSimulator:
    def __init__(self):
        self.behaviors = {
            'hesitant': {'pause_probability': 0.3, 'back_probability': 0.2},
            'confident': {'pause_probability': 0.1, 'back_probability': 0.05},
            'mobile': {'typing_speed': 0.7, 'error_rate': 0.15},
            'desktop': {'typing_speed': 1.0, 'error_rate': 0.08}
        }

class MultiAgentCheckout:
    def __init__(self):
        self.agents = {
            'cart_specialist': CheckoutAgent(focus='cart_optimization'),
            'payment_specialist': CheckoutAgent(focus='payment_optimization'),
            'shipping_specialist': CheckoutAgent(focus='shipping_optimization')
        }

class CheckoutBenchmark:
    def __init__(self):
        self.industry_benchmarks = {
            'avg_completion_time': 180,  # seconds
            'error_rate': 0.05,
            'cart_abandonment': 0.69,
            'mobile_conversion': 0.15
        }

if __name__ == "__main__":
    # Example usage with your checkout agent
    from checkout_agent import CheckoutAgent
    
    agent = CheckoutAgent()
    results = agent.simulate_checkout()
    
    analyzer = CheckoutAnalyzer()
    analysis = analyzer.analyze_checkout(results)
    print(analyzer.generate_report(analysis)) 
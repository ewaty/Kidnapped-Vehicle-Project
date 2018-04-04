/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 25;
	// Normal (Gaussian) distribution for x, y and theta.
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	// Generate particles around GPS provided coordinates
	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p.id = i;
		p.weight = 1;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		particles.push_back(p);
	}
	
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Normal (Gaussian) distribution for x, y and theta.
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	// Check if yaw rate greater than 0, update particles with predicted position and angle, add gaussian noise
	if (fabs(yaw_rate) > 0.001){
		for (std::vector<Particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
			(*it).x += velocity/yaw_rate*(sin((*it).theta + yaw_rate*delta_t) - sin((*it).theta)) + dist_x(gen);
			(*it).y += velocity/yaw_rate*(cos((*it).theta) - cos((*it).theta + yaw_rate*delta_t)) + dist_y(gen);
			(*it).theta += yaw_rate*delta_t + dist_theta(gen);
		}	
	}
	// If yaw rate close to 0, update particles with predicted position and angle, add gaussian noise with appropriate modifications
	else {
		for (std::vector<Particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
			(*it).x += velocity*delta_t*cos((*it).theta) + dist_x(gen);
			(*it).y += velocity*delta_t*sin((*it).theta) + dist_y(gen);
			(*it).theta += dist_theta(gen);	
		}
	}
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& observations, double std_landmark[], const Map &map_landmarks) {
	double obs_x = 0; //
	double obs_y = 0; //
	double h_x = 0;   // 
	double h_y = 0;   // helper 
	double h_cos = 0; // variables
	double h_sin = 0; //
	double dist = 0;  //
	double cons = 1/(2*M_PI*std_landmark[0]*std_landmark[1]); //
	// Iterate over all particles
	for (std::vector<Particle>::iterator it = particles.begin(); it!= particles.end(); ++it){
		// Fill helper variables		
		h_x = (*it).x;
		h_y = (*it).y;
		h_cos = cos((*it).theta);
		h_sin = sin((*it).theta);
		// Clear vectors from previous values, they will be refilled later
		(*it).associations.clear();
		(*it).sense_x.clear();
		(*it).sense_y.clear();
		(*it).weight = 1;
		// Iterate over all observations
		for (std::vector<LandmarkObs>::const_iterator jt = observations.begin(); jt != observations.end(); ++jt){
			// Find predicted position of measurement from particle
			obs_x = h_x + h_cos*((*jt).x) - h_sin*((*jt).y);
			obs_y = h_y + h_sin*((*jt).x) + h_cos*((*jt).y);
			int counter = 0;
			int position = 0;
			double distance = 9999999999999;
			// Iterate over all landmarks
			for (std::vector<Map::single_landmark_s>::const_iterator lt = map_landmarks.landmark_list.begin(); lt != map_landmarks.landmark_list.end(); ++lt){
				// Find landmark whose distance to predicted measurement is closest
				dist = sqrt(pow((obs_x - (*lt).x_f),2) + pow((obs_y - (*lt).y_f),2));
				if (dist < distance){
					distance = dist;
					position = counter;
				}
				counter += 1;
			}
			// Update weights, associations, sense_x and sense_y
			(*it).weight *= cons*exp(-(pow(obs_x - map_landmarks.landmark_list[position].x_f ,2)/(2*pow(std_landmark[0], 2)) + pow(obs_y - map_landmarks.landmark_list[position].y_f ,2)/(2*pow(std_landmark[1], 2))));
			(*it).associations.push_back(map_landmarks.landmark_list[position].id_i);
			(*it).sense_x.push_back(map_landmarks.landmark_list[position].x_f);
			(*it).sense_y.push_back(map_landmarks.landmark_list[position].y_f);
		}
		// Save calculated weight in weights vector
		weights.push_back((*it).weight);	
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Call dataAssociation function which also updates weights 
	ParticleFilter::dataAssociation(observations, std_landmark, map_landmarks);	
}

void ParticleFilter::resample() {
	// Generate particle indexes with probabilities proportional to particle weights
	std::default_random_engine generator;
	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::vector<Particle> temp_particle;
	// Generate new particles
	for (int i=0; i<num_particles; ++i){
		int number = d(generator);
		temp_particle.push_back(particles[number]);
		temp_particle[i].id = i;
	}
	particles.clear();
	weights.clear();
	// Overwrite the particles vector with new particles
	particles = temp_particle;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

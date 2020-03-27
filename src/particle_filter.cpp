/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::normal_distribution<double> gaussian(0.0, 1.0);
  particles.clear();
  for (int i = 0; i < num_particles; ++i) {
      particles.emplace_back();
      particles.back().x = x + std[0] * gaussian(generator);
      particles.back().y = y + std[1] * gaussian(generator);
      particles.back().theta = theta + std[2] * gaussian(generator);
      particles.back().weight = 1.0;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    std::normal_distribution<double> gaussian(0.0, 1.0);
    if (yaw_rate > 0.001 || yaw_rate < -0.001) {
        for (auto& p: particles) {
            p.x += std_pos[0] * gaussian(generator) +
            (velocity/yaw_rate)*(sin(p.theta + delta_t*yaw_rate) - sin(p.theta));
            p.y += std_pos[1] * gaussian(generator) +
            (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + delta_t*yaw_rate));
            p.theta += std_pos[2] * gaussian(generator) + delta_t * yaw_rate;
        }
    } else {
        for (auto& p: particles) {
            p.x += std_pos[0]*gaussian(generator) + delta_t*velocity*cos(p.theta);
            p.y += std_pos[1]*gaussian(generator) + delta_t*velocity*sin(p.theta);
            p.theta += std_pos[2]*gaussian(generator);
        }
    }
}

// Computes the PDF of a 2D Gaussian evaluated at x, y, where the Gaussian
// has mean (mx, my) and covariance ((sx, 0), (0, sy))
double GaussianPDF2D(double x, double y, double mx, double my, double sx,
        double sy) {
    double pdf = std::exp(-(x-mx)*(x-mx)/(2*sx*sx) -(y-my)*(y-my)/(2*sy*sy))
        / (2*M_PI*sx*sy);
    return pdf;
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    // Perform data association
    for (auto& particle: particles) {
        // Reset particle frame landmark locations
        particle.associations.clear();
        particle.sense_x.clear();
        particle.sense_y.clear();
        double ct = std::cos(particle.theta);
        double st = std::sin(particle.theta);
        for (auto & observation: observations) {
            // Transform observation to map coordinates
            particle.associations.push_back(-1);
            particle.sense_x.push_back(
                    ct * observation.x - st * observation.y + particle.x);
            particle.sense_y.push_back(
                    st * observation.x + ct * observation.y + particle.y);
            int best_landmark = -1;
            double best_distance = std::numeric_limits<double>::max();
            // Find associations
            for (unsigned int l_i = 0; l_i < map_landmarks.landmark_list.size();
                    ++l_i) {
                double l_x = map_landmarks.landmark_list[l_i].x_f;
                double l_y = map_landmarks.landmark_list[l_i].y_f;
                if (dist(particle.x, particle.y, l_x, l_y) > sensor_range)
                    continue;
                double distance = dist(particle.sense_x.back(),
                        particle.sense_y.back(), l_x, l_y);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_landmark = l_i;
                }
            }
            // To deal with 1 based indexing
            particle.associations.back() = best_landmark + 1;
        }
        // Update weight
        for (unsigned int obs_i = 0; obs_i < particle.associations.size();
                ++obs_i) {
            int association = particle.associations[obs_i];
            if (association == -1) continue;
            association -= 1;
            double l_x = map_landmarks.landmark_list[association].x_f;
            double l_y = map_landmarks.landmark_list[association].y_f;
            double likelihood = GaussianPDF2D(particle.sense_x[obs_i],
                                              particle.sense_y[obs_i],
                                              l_x, l_y,
                                              std_landmark[0],
                                              std_landmark[1]);
            particle.weight *= likelihood;
        }
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    // Normalize particle weights
    double total_weight = 0;
    for (auto& particle: particles) {
        total_weight += particle.weight;
    }
    for (auto& particle: particles) {
        particle.weight /= total_weight;
    }
    // Perform roulette resampling
    std::uniform_real_distribution<double> distribution(0, 1);
    double increment = 1.0/particles.size();
    double target_pos = distribution(generator);
    double current_pos = 0;
    unsigned int current_particle = 0;
    std::vector<unsigned int> indices;
    for (unsigned int i = 0; i < particles.size(); ++i) {
        while (current_pos + particles[current_particle].weight < target_pos) {
            current_pos += particles[current_particle].weight;
            current_particle++;
            if (current_particle >= particles.size()) {
                current_particle = 0;
                current_pos -= 1.0;
                target_pos -= 1.0;
            }
        }
        indices.push_back(current_particle);
        target_pos += increment;
    }
    std::vector<Particle> new_particles;
    for (auto& resampled_index: indices) {
        new_particles.push_back(particles[resampled_index]);
        new_particles.back().weight = 1.0;
    }
    particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

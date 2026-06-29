#include "Garfield/ComponentTcadBase.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>

#include "Garfield/Exceptions.hh"
#include "Garfield/GarfieldConstants.hh"
#include "Garfield/Medium.hh"
#include "Garfield/Utilities.hh"

namespace {

bool ExtractFromSquareBrackets(std::string& line) {
  const auto bra = line.find('[');
  const auto ket = line.find(']');
  if (ket < bra || bra == std::string::npos || ket == std::string::npos) {
    return false;
  }
  line = line.substr(bra + 1, ket - bra - 1);
  return true;
}

bool ExtractFromBrackets(std::string& line) {
  const auto bra = line.find('(');
  const auto ket = line.find(')');
  if (ket < bra || bra == std::string::npos || ket == std::string::npos) {
    return false;
  }
  line = line.substr(bra + 1, ket - bra - 1);
  return true;
}

int GetTrapIndex(const std::string& f) {
  const std::string s = "TrapOccupation_";
  if (!Garfield::startsWith(f, s)) return -1;
  const auto n0 = s.size();
  const auto n1 = f.find('(');
  if (n1 < n0 || n1 == std::string::npos) return -1;
  return std::stoi(f.substr(n0, n1 - n0));
}

void PrintError(const std::string& fcn, const std::string& filename,
                const std::size_t line) {
  std::cerr << fcn << ":\n"
            << "    Error reading file " << filename << " (line " << line
            << ").\n";
}
}  // namespace

namespace Garfield {

template <size_t N>
void ComponentTcadBase<N>::WeightingField(const double x, const double y,
                                          const double z, double& wx,
                                          double& wy, double& wz,
                                          const std::string& label) {
  wx = wy = wz = 0.;
  if (m_wfield[label].empty()) {
    std::cerr << m_className << "::WeightingField: No fieldmap for " << label
              << " available.\n";
    return;
  }
  double dx = 0., dy = 0., dz = 0.;
  if (!GetOffset(label, dx, dy, dz)) return;
  Interpolate(x - dx, y - dy, z - dz, m_wfield[label], wx, wy, wz);
}

template <size_t N>
double ComponentTcadBase<N>::WeightingPotential(const double xin,
                                                const double yin,
                                                const double zin,
                                                const std::string& label0) {
  if (!m_ready) return 0.0;

  // Resolve copies
  std::string label = label0;
  double xq[3] = {xin, yin, zin};

  auto itCopy = m_wfieldCopies.find(label0);
  if (itCopy != m_wfieldCopies.end()) {
    const WeightingFieldCopy& w = itCopy->second;
    label = w.source;

    double in[3] = {xin, yin, zin};
    double out[3] = {0., 0., 0.};

    for (size_t r = 0; r < N; ++r) {
      out[r] = w.rot[r][0] * in[0] + w.rot[r][1] * in[1] + w.rot[r][2] * in[2] +
               w.trans[r];
    }
    xq[0] = out[0];
    if constexpr (N >= 2) xq[1] = out[1];
    if constexpr (N >= 3) xq[2] = out[2];
  }

  if (m_wpot.count(label) == 0 || m_wpot[label].empty()) {
    std::cerr << m_className << "::WeightingPotential: No fieldmap for "
              << label << " available.\n";
    return 0.;
  }

  double dx = 0.0, dy = 0.0, dz = 0.0;
  if (!GetOffset(label, dx, dy, dz)) return 0.0;

  double v = 0.0;
  if constexpr (N == 2) {
    Interpolate(xq[0] - dx, xq[1] - dy, 0.0, m_wpot[label], v);
  } else {
    Interpolate(xq[0] - dx, xq[1] - dy, xq[2] - dz, m_wpot[label], v);
  }
  return v;
}

template <size_t N>
void ComponentTcadBase<N>::DelayedWeightingField(const double x, const double y,
                                                 const double z, const double t,
                                                 double& wx, double& wy,
                                                 double& wz,
                                                 const std::string& label) {
  wx = wy = wz = 0.;

  if (m_dwf[label].empty()) {
    std::cerr << m_className << "::DelayedWeightingField: No fieldmap for "
              << label << " available.\n";
    return;
  }

  if (m_dwtf[label].empty()) return;
  if (t < m_dwtf[label].front() || t > m_dwtf[label].back()) return;

  double dx = 0., dy = 0., dz = 0.;
  if (!GetOffset(label, dx, dy, dz)) return;

  const auto it1 =
      std::upper_bound(m_dwtf[label].cbegin(), m_dwtf[label].cend(), t);
  const auto it0 = std::prev(it1);
  const double dt = t - *it0;
  const auto i0 = std::distance(m_dwtf[label].cbegin(), it0);
  double wx0 = 0., wy0 = 0., wz0 = 0.;
  Interpolate(x - dx, y - dy, z - dz, m_dwf[label][i0], wx0, wy0, wz0);
  if (dt < Small || it1 == m_dwtf[label].cend()) {
    wx = wx0;
    wy = wy0;
    wz = wz0;
    return;
  }
  const auto i1 = std::distance(m_dwtf[label].cbegin(), it1);
  double wx1 = 0., wy1 = 0., wz1 = 0.;
  Interpolate(x - dx, y - dy, z - dz, m_dwf[label][i1], wx1, wy1, wz1);
  const double f1 = dt / (*it1 - *it0);
  const double f0 = 1. - f1;
  wx = f0 * wx0 + f1 * wx1;
  wy = f0 * wy0 + f1 * wy1;
  wz = f0 * wz0 + f1 * wz1;
}

template <size_t N>
double ComponentTcadBase<N>::DelayedWeightingPotential(
    const double x, const double y, const double z, const double t,
    const std::string& label) {
  if (m_dwp[label].empty()) {
    std::cerr << m_className << "::DelayedWeightingPotential: No fieldmap for "
              << label << " available.\n";
    return 0.;
  }

  if (m_dwtp[label].empty()) return 0.;
  if (t < m_dwtp[label].front() || t > m_dwtp[label].back()) return 0.;

  double dx = 0., dy = 0., dz = 0.;
  if (!GetOffset(label, dx, dy, dz)) return 0.;

  const auto it1 =
      std::upper_bound(m_dwtp[label].cbegin(), m_dwtp[label].cend(), t);
  const auto it0 = std::prev(it1);
  const double dt = t - *it0;
  const auto i0 = std::distance(m_dwtp[label].cbegin(), it0);
  double v0 = 0.;
  Interpolate(x - dx, y - dy, z - dz, m_dwp[label][i0], v0);
  if (dt < Small || it1 == m_dwtp[label].cend()) return v0;

  const auto i1 = std::distance(m_dwtp[label].cbegin(), it1);
  double v1 = 0.;
  Interpolate(x - dx, y - dy, z - dz, m_dwp[label][i1], v1);
  const double f1 = dt / (*it1 - *it0);
  const double f0 = 1. - f1;
  return f0 * v0 + f1 * v1;
}

template <size_t N>
bool ComponentTcadBase<N>::GetOffset(const std::string& label, double& dx,
                                     double& dy, double& dz) const {
  if (m_wshift.count(label) == 0 || m_wshift.at(label).empty()) return false;

  dx = m_wshift.at(label)[0];
  dy = m_wshift.at(label)[1];
  dz = m_wshift.at(label)[2];

  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::Initialise(const std::string& gridfilename,
                                      const std::string& datafilename) {
  m_ready = false;
  Cleanup();
  // Import mesh data from .grd file.
  if (!LoadGrid(gridfilename)) {
    std::cerr << m_className << "::Initialise:\n"
              << "    Importing mesh data failed.\n";
    Cleanup();
    return false;
  }
  // Import electric field, potential and other data from .dat file.
  if (!LoadData(datafilename)) {
    std::cerr << m_className << "::Initialise:\n"
              << "    Importing electric field and potential failed.\n";
    Cleanup();
    return false;
  }

  // Find min./max. coordinates and potentials.
  for (size_t i = 0; i < N; ++i) {
    m_bbMax[i] = m_vertices[m_elements[0].vertex[0]][i];
    m_bbMin[i] = m_bbMax[i];
  }
  const size_t nElements = m_elements.size();
  for (size_t i = 0; i < nElements; ++i) {
    Element& element = m_elements[i];
    std::array<double, N> xmin = m_vertices[element.vertex[0]];
    std::array<double, N> xmax = m_vertices[element.vertex[0]];
    const auto nV = ElementVertices(m_elements[i]);
    for (std::size_t j = 0; j < nV; ++j) {
      const auto& v = m_vertices[m_elements[i].vertex[j]];
      for (size_t k = 0; k < N; ++k) {
        xmin[k] = std::min(xmin[k], v[k]);
        xmax[k] = std::max(xmax[k], v[k]);
      }

      double distMin = std::numeric_limits<double>::max();
      for (size_t k = 0; k < j; ++k) {
        const auto& vComp = m_vertices[m_elements[i].vertex[k]];
        double distV = 0.;

        for (size_t l = 0; l < N; ++l)
          distV += (v[l] - vComp[l]) * (v[l] - vComp[l]);
        distV = std::sqrt(distV);
        distMin = std::min(distMin, distV);
      }
      m_elements[i].length = distMin;
    }
    constexpr double tol = 1.e-6;
    for (size_t k = 0; k < N; ++k) {
      m_elements[i].bbMin[k] = xmin[k] - tol;
      m_elements[i].bbMax[k] = xmax[k] + tol;
      m_bbMin[k] = std::min(m_bbMin[k], xmin[k]);
      m_bbMax[k] = std::max(m_bbMax[k], xmax[k]);
    }
  }
  m_pMin = *std::min_element(m_epot.begin(), m_epot.end());
  m_pMax = *std::max_element(m_epot.begin(), m_epot.end());

  std::cout << m_className << "::Initialise:\n"
            << "    Available data:\n";
  if (!m_epot.empty()) std::cout << "      Electrostatic potential\n";
  if (!m_efield.empty()) std::cout << "      Electric field\n";
  if (!m_eMobility.empty()) std::cout << "      Electron mobility\n";
  if (!m_hMobility.empty()) std::cout << "      Hole mobility\n";
  if (!m_eVelocity.empty()) std::cout << "      Electron velocity\n";
  if (!m_hVelocity.empty()) std::cout << "      Hole velocity\n";
  if (!m_eAlpha.empty()) std::cout << "      Electron impact ionisation\n";
  if (!m_hAlpha.empty()) std::cout << "      Hole impact ionisation\n";
  if (!m_eLifetime.empty()) std::cout << "      Electron lifetime\n";
  if (!m_hLifetime.empty()) std::cout << "      Hole lifetime\n";
  if (!m_defects.empty()) {
    std::cout << "      Occupancies for " << m_defects.size() << " traps\n";
  }
  const std::array<std::string, 3> axes = {"x", "y", "z"};
  std::cout << "    Bounding box:\n";
  for (size_t i = 0; i < N; ++i) {
    std::cout << "      " << m_bbMin[i] << " < " << axes[i] << " [cm] < "
              << m_bbMax[i] << "\n";
  }
  std::cout << "    Voltage range:\n"
            << "      " << m_pMin << " < V < " << m_pMax << "\n";

  bool ok = true;

  // Count the number of elements belonging to a region.
  const auto nRegions = m_regions.size();
  std::vector<size_t> nElementsByRegion(nRegions, 0);
  // Keep track of elements that are not part of any region.
  std::vector<size_t> looseElements;

  // Count the different element shapes.
  std::map<int, std::size_t> nElementsByShape;
  if (N == 2) {
    nElementsByShape = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
  } else {
    nElementsByShape = {{0, 0}, {2, 0}, {5, 0}};
  }
  std::size_t nElementsOther = 0;

  // Keep track of degenerate elements.
  std::vector<size_t> degenerateElements;

  for (size_t i = 0; i < nElements; ++i) {
    const Element& element = m_elements[i];
    if (element.region < nRegions) {
      ++nElementsByRegion[element.region];
    } else {
      looseElements.push_back(i);
    }
    if (nElementsByShape.count(element.type) == 0) {
      ++nElementsOther;
      continue;
    }
    nElementsByShape[element.type] += 1;
    bool degenerate = false;
    const auto nV = ElementVertices(m_elements[i]);
    for (std::size_t j = 0; j < nV; ++j) {
      for (std::size_t k = j + 1; k < nV; ++k) {
        if (element.vertex[j] == element.vertex[k]) {
          degenerate = true;
          break;
        }
      }
      if (degenerate) break;
    }
    if (degenerate) {
      degenerateElements.push_back(i);
    }
  }

  if (!degenerateElements.empty()) {
    std::cerr << m_className << "::Initialise:\n"
              << "    The following elements are degenerate:\n";
    for (size_t i : degenerateElements) std::cerr << "      " << i << "\n";
    ok = false;
  }

  if (!looseElements.empty()) {
    std::cerr << m_className << "::Initialise:\n"
              << "    The following elements are not part of any region:\n";
    for (size_t i : looseElements) std::cerr << "      " << i << "\n";
    ok = false;
  }

  std::cout << m_className << "::Initialise:\n"
            << "    Number of regions: " << nRegions << "\n";
  for (size_t i = 0; i < nRegions; ++i) {
    std::cout << "      " << i << ": " << m_regions[i].name;
    if (!m_regions[i].material.empty()) {
      std::cout << " (" << m_regions[i].material << ")";
    }
    std::cout << ", " << nElementsByRegion[i] << " elements\n";
  }

  std::map<int, std::string> shapes = {{0, "points"},
                                       {1, "lines"},
                                       {2, "triangles"},
                                       {3, "rectangles"},
                                       {5, "tetrahedra"}};

  std::cout << "    Number of elements: " << nElements << "\n";
  for (const auto& n : nElementsByShape) {
    if (n.second > 0) {
      std::cout << "      " << n.second << " " << shapes[n.first] << "\n";
    }
  }
  if (nElementsOther > 0) {
    std::cerr << "      " << nElementsOther << " elements of unknown type.\n"
              << "      Program bug!\n";
    m_ready = false;
    Cleanup();
    return false;
  }

  std::cout << "    Number of vertices: " << m_vertices.size() << "\n";
  if (!ok) {
    m_ready = false;
    Cleanup();
    return false;
  }

  FillTree();

  m_ready = true;
  UpdatePeriodicity();
  std::cout << m_className << "::Initialise: Initialisation finished.\n";
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::GetVoltageRange(double& vmin, double& vmax) {
  if (!m_ready) return false;
  vmin = m_pMin;
  vmax = m_pMax;
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::SetWeightingField(const std::string& datfile1,
                                             const std::string& datfile2,
                                             const double dv,
                                             const std::string& label) {
  if (!m_ready) {
    std::cerr << m_className << "::SetWeightingField:\n"
              << "    Mesh is not available. Call Initialise first.\n";
    return false;
  }
  if (dv < Small) {
    std::cerr << m_className << "::SetWeightingField:\n"
              << "    Voltage difference must be > 0.\n";
    return false;
  }
  const double s = 1. / dv;

  // Check if a weighting field with the same label already exists.
  if (m_wfield.count(label) > 0) {
    std::cout << m_className << "::SetWeightingField:\n"
              << "    Replacing existing weighting field " << label << ".\n";
    m_wfield[label].clear();
    m_wpot[label].clear();
    m_wshift[label].clear();
  }
  if (m_dwf.count(label) > 0) {
    m_dwf[label].clear();
    m_dwp[label].clear();

    m_dwtf.clear();
    m_dwtp.clear();
  }

  // Load first the field/potential at nominal bias.
  std::vector<std::array<double, N> > wf1;
  std::vector<double> wp1;
  if (!LoadWeightingField(datfile1, wf1, wp1)) {
    std::cerr << m_className << "::SetWeightingField:\n"
              << "    Could not import data from " << datfile1 << ".\n";
    return false;
  }

  // Then load the field/potential for the configuration with the potential
  // at the electrode to be read out increased by small voltage dv.
  std::vector<std::array<double, N> > wf2;
  std::vector<double> wp2;
  if (!LoadWeightingField(datfile2, wf2, wp2)) {
    std::cerr << m_className << "::SetWeightingField:\n"
              << "    Could not import data from " << datfile2 << ".\n";
    return false;
  }
  const size_t nVertices = m_vertices.size();
  bool foundField = true;
  if (wf1.size() != nVertices || wf2.size() != nVertices) {
    foundField = false;
    std::cerr << m_className << "::SetWeightingField:\n"
              << "    Could not load electric field values.\n";
  }
  bool foundPotential = true;
  if (wp1.size() != nVertices || wp2.size() != nVertices) {
    foundPotential = false;
    std::cerr << m_className << "::SetWeightingField:\n"
              << "    Could not load electrostatic potentials.\n";
  }
  if (!foundField && !foundPotential) return false;
  if (foundField) {
    m_wfield[label].resize(nVertices);
    for (size_t i = 0; i < nVertices; ++i) {
      for (size_t j = 0; j < N; ++j) {
        m_wfield[label][i][j] = (wf2[i][j] - wf1[i][j]) * s;
      }
    }
  }
  if (foundPotential) {
    m_wpot[label].assign(nVertices, 0.);
    for (size_t i = 0; i < nVertices; ++i) {
      m_wpot[label][i] = (wp2[i] - wp1[i]) * s;
    }
  }

  m_wshift[label] = {0., 0., 0.};
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::SetDynamicWeightingPotential(
    const std::string& datfile1, const std::string& datfile2, const double dv,
    const double t, const std::string& label) {
  if (t < Small) {
    return SetWeightingField(datfile1, datfile2, dv, label);
  }

  if (!m_ready) {
    std::cerr << m_className << "::SetDynamicWeightingPotential:\n"
              << "    Mesh is not available. Call Initialise first.\n";
    return false;
  }
  if (dv < Small) {
    std::cerr << m_className << "::SetDynamicWeightingPotential:\n"
              << "    Voltage difference must be > 0.\n";
    return false;
  }
  const double s = 1. / dv;

  // Check if the prompt weighting potential with the same label already exists.
  if (m_wpot.count(label) == 0 || m_wpot[label].empty()) {
    std::cerr << m_className << "::SetDynamicWeightingPotential:\n"
              << "    Prompt component not present.\n"
              << "    Import the map for t = 0 first.\n";
    return false;
  }

  // Load the first map.
  std::vector<std::array<double, N> > wf1;
  std::vector<double> wp1;
  if (!LoadWeightingField(datfile1, wf1, wp1)) {
    std::cerr << m_className << "::SetDynamicWeightingPotential:\n"
              << "    Could not import data from " << datfile1 << ".\n";
    return false;
  }
  // Load the second map.
  std::vector<std::array<double, N> > wf2;
  std::vector<double> wp2;
  if (!LoadWeightingField(datfile2, wf2, wp2)) {
    std::cerr << m_className << "::SetDynamicWeightingPotential:\n"
              << "    Could not import data from " << datfile2 << ".\n";
    return false;
  }
  const size_t nVertices = m_vertices.size();
  if (wp1.size() != nVertices || wp2.size() != nVertices) {
    std::cerr << m_className << "::SetDynamicWeightingPotential:\n"
              << "    Could not load electrostatic potentials.\n";
    return false;
  }
  if (m_wpot[label].size() != nVertices) {
    std::cerr << m_className << "::SetDynamicWeightingPotential:\n"
              << "    Prompt weighting potential not present.\n";
    return false;
  }
  std::vector<double> wp(nVertices, 0.);
  for (size_t i = 0; i < nVertices; ++i) {
    wp[i] = (wp2[i] - wp1[i]) * s;
    // Subtract the prompt component.
    wp[i] -= m_wpot[label][i];
  }
  if (m_dwtp[label].empty() || t > m_dwtp[label].back()) {
    m_dwtp[label].push_back(t);
    m_dwp[label].push_back(std::move(wp));
  } else {
    const auto it =
        std::upper_bound(m_dwtp[label].begin(), m_dwtp[label].end(), t);
    const auto n = std::distance(m_dwtp[label].begin(), it);
    m_dwtp[label].insert(it, t);
    m_dwp[label].insert(m_dwp[label].begin() + n, std::move(wp));
  }
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::SetDynamicWeightingField(const std::string& datfile1,
                                                    const std::string& datfile2,
                                                    const double dv,
                                                    const double t,
                                                    const std::string& label) {
  if (t < Small) {
    return SetWeightingField(datfile1, datfile2, dv, label);
  }
  if (!m_ready) {
    std::cerr << m_className << "::SetDynamicWeightingField:\n"
              << "    Mesh is not available. Call Initialise first.\n";
    return false;
  }
  if (dv < Small) {
    std::cerr << m_className << "::SetDynamicWeightingField:\n"
              << "    Voltage difference must be > 0.\n";
    return false;
  }
  const double s = 1. / dv;

  // Check if the prompt weighting potential with the same label already exists.
  if (m_wfield.count(label) == 0 || m_wfield[label].empty()) {
    std::cerr << m_className << "::SetDynamicWeightingField:\n"
              << "    Prompt component not present.\n"
              << "    Import the map for t = 0 first.\n";
    return false;
  }

  // Load the first map.
  std::vector<std::array<double, N> > wf1;
  std::vector<double> wp1;
  if (!LoadWeightingField(datfile1, wf1, wp1)) {
    std::cerr << m_className << "::SetDynamicWeightingField:\n"
              << "    Could not import data from " << datfile1 << ".\n";
    return false;
  }
  // Load the second map.
  std::vector<std::array<double, N> > wf2;
  std::vector<double> wp2;
  if (!LoadWeightingField(datfile2, wf2, wp2)) {
    std::cerr << m_className << "::SetDynamicWeightingField:\n"
              << "    Could not import data from " << datfile2 << ".\n";
    return false;
  }
  const size_t nVertices = m_vertices.size();
  if (wf1.size() != nVertices || wf2.size() != nVertices) {
    std::cerr << m_className << "::SetDynamicWeightingField:\n"
              << "    Could not load electric field values.\n";
    return false;
  }
  if (m_wfield[label].size() != nVertices) {
    std::cerr << m_className << "::SetDynamicWeightingField:\n"
              << "    Prompt weighting field not present.\n";
    return false;
  }
  std::vector<std::array<double, N> > wf;
  wf.resize(nVertices);
  for (size_t i = 0; i < nVertices; ++i) {
    for (size_t j = 0; j < N; ++j) {
      wf[i][j] = (wf2[i][j] - wf1[i][j]) * s;
    }
  }
  if (m_dwtf[label].empty() || t > m_dwtf[label].back()) {
    m_dwtf[label].push_back(t);
    m_dwf[label].push_back(std::move(wf));
  } else {
    const auto it =
        std::upper_bound(m_dwtf[label].begin(), m_dwtf[label].end(), t);
    const auto n = std::distance(m_dwtf[label].begin(), it);
    m_dwtf[label].insert(it, t);
    m_dwf[label].insert(m_dwf[label].begin() + n, std::move(wf));
  }
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::SetWeightingFieldShift(const std::string& label,
                                                  const double x,
                                                  const double y,
                                                  const double z) {
  if ((m_wfield.count(label) == 0 || m_wfield[label].empty()) &&
      (m_wpot.count(label) == 0 || m_wpot[label].empty())) {
    std::cerr << m_className << "::SetWeightingFieldShift:\n"
              << "    No map of weighting potentials/fields loaded.\n";
    return false;
  }

  m_wshift[label] = {x, y, z};

  if (m_wshift.count(label) > 0) {
    std::cout << m_className << "::SetWeightingFieldShift:\n"
              << "    Changing offset of electrode \'" << label << "\' to ("
              << x << ", " << y << ", " << z << ") cm.\n";

  } else {
    std::cout << m_className << "::SetWeightingFieldShift:\n"
              << "    Adding electrode \'" << label << "\' with offset (" << x
              << ", " << y << ", " << z << ") cm.\n";
  }
  return true;
}

template <size_t N>
void ComponentTcadBase<N>::EnableVelocityMap(const bool on) {
  m_useVelocityMap = on;
  if (on && m_ready && (m_eVelocity.empty() && m_hVelocity.empty())) {
    std::cout << m_className << "::EnableVelocityMap:\n"
              << "    Warning: current map does not have velocity data.\n";
  }
}

template <size_t N>
void ComponentTcadBase<N>::EnableTrapOccupationMap(const bool on) {
  m_useTrapOccMap = on;
  if (on && m_ready && m_defectOcc.empty()) {
    std::cout << m_className << "::EnableTrapOccupationMap:\n"
              << "    Warning: current map does not have "
              << " trap occupation probability data.\n";
  }
  UpdateAttachment();
}

template <size_t N>
void ComponentTcadBase<N>::EnableLifetimeMap(const bool on) {
  m_useLifetimeMap = on;
  if (on && m_ready && (m_eLifetime.empty() && m_hLifetime.empty())) {
    std::cout << m_className << "::EnableLifetimeMap:\n"
              << "    Warning: current map does not have lifetime data.\n";
  }
  UpdateAttachment();
}

template <size_t N>
bool ComponentTcadBase<N>::GetElementNodes(const size_t i,
                                           std::vector<size_t>& nodes, bool allNodes) const {
  nodes.clear();
  if (i >= m_elements.size()) {
    std::cerr << m_className << "::GetElementNodes: Index out of range.\n";
    return false;
  }

  const Element& element = m_elements[i];
  const size_t nVertices = ElementVertices(element);
  for (size_t j = 0; j < nVertices; ++j) {
    nodes.push_back(element.vertex[j]);
  }
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::GetElementRegion(const size_t i, size_t& region,
                                            bool& drift) const {
  if (i >= m_elements.size()) {
    std::cerr << m_className << "::GetElementRegion: Index out of range.\n";
    return false;
  }
  region = m_elements[i].region;
  drift = m_regions[region].drift;
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::LoadGrid(const std::string& filename) {
  // Open the file containing the mesh description.
  std::ifstream gridfile(filename);
  if (!gridfile) {
    std::cerr << m_className << "::LoadGrid:\n"
              << "    Could not open file " << filename << ".\n";
    return false;
  }
  // Delete existing mesh information.
  Cleanup();
  // Count line numbers.
  std::size_t iLine = 0;
  // Get the number of regions.
  size_t nRegions = 0;
  // Read the file line by line.
  std::string line;
  while (std::getline(gridfile, line)) {
    ++iLine;
    // Strip white space from the beginning of the line.
    ltrim(line);
    if (line.empty()) continue;
    // Find entry 'nb_regions'.
    if (line.substr(0, 10) != "nb_regions") continue;
    const auto pEq = line.find('=');
    if (pEq == std::string::npos) {
      // No "=" sign found.
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Could not read number of regions.\n";
      return false;
    }
    line = line.substr(pEq + 1);
    std::istringstream data(line);
    data >> nRegions;
    break;
  }
  if (gridfile.eof()) {
    // Reached end of file.
    std::cerr << m_className << "::LoadGrid:\n"
              << "    Could not find entry 'nb_regions' in file\n"
              << "    " << filename << ".\n";
    return false;
  } else if (gridfile.fail()) {
    // Error reading from the file.
    PrintError(m_className + "::LoadGrid", filename, iLine);
    return false;
  }
  m_regions.resize(nRegions);
  for (size_t j = 0; j < nRegions; ++j) {
    m_regions[j].name = "";
    m_regions[j].material = "";
    m_regions[j].drift = false;
    m_regions[j].medium = nullptr;
  }
  if (m_debug) {
    std::cout << m_className << "::LoadGrid:\n"
              << "    Found " << nRegions << " regions.\n";
  }
  // Get the region names.
  while (std::getline(gridfile, line)) {
    ++iLine;
    ltrim(line);
    if (line.empty()) continue;
    // Find entry 'regions'.
    if (line.substr(0, 7) != "regions") continue;
    // Get region names (given in brackets).
    if (!ExtractFromSquareBrackets(line)) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Could not read region names.\n";
      return false;
    }
    std::istringstream data(line);
    for (size_t j = 0; j < nRegions; ++j) {
      data >> m_regions[j].name;
      data.clear();
      // Assume by default that all regions are active.
      m_regions[j].drift = true;
      m_regions[j].medium = nullptr;
    }
    break;
  }
  if (gridfile.eof()) {
    // Reached end of file.
    std::cerr << m_className << "::LoadGrid:\n"
              << "    Could not find entry 'regions' in file\n"
              << "    " << filename << ".\n";
    return false;
  } else if (gridfile.fail()) {
    // Error reading from the file.
    PrintError(m_className + "::LoadGrid", filename, iLine);
    return false;
  }

  // Get the materials.
  while (std::getline(gridfile, line)) {
    ++iLine;
    ltrim(line);
    if (line.empty()) continue;
    // Find entry 'materials'.
    if (line.substr(0, 9) != "materials") continue;
    // Get region names (given in brackets).
    if (!ExtractFromSquareBrackets(line)) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Could not read materials.\n";
      return false;
    }
    std::istringstream data(line);
    for (size_t j = 0; j < nRegions; ++j) {
      data >> m_regions[j].material;
      data.clear();
    }
    break;
  }
  if (gridfile.eof()) {
    // Reached end of file.
    std::cerr << m_className << "::LoadGrid:\n"
              << "    Could not find entry 'materials' in file\n"
              << "    " << filename << ".\n";
  } else if (gridfile.fail()) {
    // Error reading from the file.
    PrintError(m_className + "::LoadGrid", filename, iLine);
  }

  // Get the vertices.
  size_t nVertices = 0;
  while (std::getline(gridfile, line)) {
    ++iLine;
    ltrim(line);
    if (line.empty()) continue;
    // Find section 'Vertices'.
    if (line.substr(0, 8) != "Vertices") continue;
    // Get number of vertices (given in brackets).
    if (!ExtractFromBrackets(line)) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Could not read number of vertices.\n";
      return false;
    }
    std::istringstream data(line);
    data >> nVertices;
    m_vertices.resize(nVertices);
    // Get the coordinates of every vertex.
    for (size_t j = 0; j < nVertices; ++j) {
      for (size_t k = 0; k < N; ++k) {
        gridfile >> m_vertices[j][k];
        // Change units from micron to cm.
        m_vertices[j][k] *= 1.e-4;
      }
      ++iLine;
    }
    break;
  }
  if (gridfile.eof()) {
    std::cerr << m_className << "::LoadGrid:\n"
              << "    Could not find section 'Vertices' in file\n"
              << "    " << filename << ".\n";
    return false;
  } else if (gridfile.fail()) {
    PrintError(m_className + "::LoadGrid", filename, iLine);
    return false;
  }

  // Get the "edges" (lines connecting two vertices).
  size_t nEdges = 0;
  // Temporary arrays for storing edge points.
  std::vector<std::size_t> edgeP1;
  std::vector<std::size_t> edgeP2;
  while (std::getline(gridfile, line)) {
    ++iLine;
    ltrim(line);
    if (line.empty()) continue;
    // Find section 'Edges'.
    if (line.substr(0, 5) != "Edges") continue;
    // Get the number of edges (given in brackets).
    if (!ExtractFromBrackets(line)) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Could not read number of edges.\n";
      return false;
    }
    std::istringstream data(line);
    data >> nEdges;
    edgeP1.resize(nEdges);
    edgeP2.resize(nEdges);
    // Get the indices of the two endpoints.
    for (size_t j = 0; j < nEdges; ++j) {
      gridfile >> edgeP1[j] >> edgeP2[j];
      ++iLine;
    }
    break;
  }
  if (gridfile.eof()) {
    std::cerr << m_className << "::LoadGrid:\n"
              << "    Could not find section 'Edges' in file\n"
              << "    " << filename << ".\n";
    return false;
  } else if (gridfile.fail()) {
    PrintError(m_className + "::LoadGrid", filename, iLine);
    return false;
  }

  for (size_t i = 0; i < nEdges; ++i) {
    // Make sure the indices of the edge endpoints are not out of range.
    if (edgeP1[i] >= nVertices || edgeP2[i] >= nVertices) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Vertex index of edge " << i << " out of range.\n";
      return false;
    }
    // Make sure the edge is non-degenerate.
    if (edgeP1[i] == edgeP2[i]) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Edge " << i << " is degenerate.\n";
      return false;
    }
  }

  // Get the "faces" (only for 3D maps).
  struct Face {
    // Indices of edges
    int edge[4];
    int type;
  };
  size_t nFaces = 0;
  std::vector<Face> faces;
  if (N == 3) {
    while (std::getline(gridfile, line)) {
      ++iLine;
      ltrim(line);
      if (line.empty()) continue;
      // Find section 'Faces'.
      if (line.substr(0, 5) != "Faces") continue;
      // Get the number of faces (given in brackets).
      if (!ExtractFromBrackets(line)) {
        std::cerr << m_className << "::LoadGrid:\n"
                  << "    Could not read number of faces.\n";
        return false;
      }
      std::istringstream data(line);
      data >> nFaces;
      faces.resize(nFaces);
      // Get the indices of the edges constituting this face.
      for (size_t j = 0; j < nFaces; ++j) {
        gridfile >> faces[j].type;
        if (faces[j].type != 3 && faces[j].type != 4) {
          std::cerr << m_className << "::LoadGrid:\n"
                    << "    Face with index " << j
                    << " has invalid number of edges, " << faces[j].type
                    << ".\n";
          return false;
        }
        for (int k = 0; k < faces[j].type; ++k) {
          gridfile >> faces[j].edge[k];
        }
      }
      iLine += nFaces - 1;
      break;
    }
    if (gridfile.eof()) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Could not find section 'Faces' in file\n"
                << "    " << filename << ".\n";
      return false;
    } else if (gridfile.fail()) {
      PrintError(m_className + "::LoadGrid", filename, iLine);
      return false;
    }
  }

  // Get the elements.
  size_t nElements = 0;
  while (std::getline(gridfile, line)) {
    ++iLine;
    ltrim(line);
    if (line.empty()) continue;
    // Find section 'Elements'.
    if (line.substr(0, 8) != "Elements") continue;
    // Get number of elements (given in brackets).
    if (!ExtractFromBrackets(line)) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Could not read number of elements.\n";
      return false;
    }
    std::istringstream data(line);
    data >> nElements;
    data.clear();
    // Resize the list of elements.
    m_elements.resize(nElements);
    // Get type and constituting edges of each element.
    for (size_t j = 0; j < nElements; ++j) {
      ++iLine;
      std::size_t type = 0;
      gridfile >> type;
      if (N == 2) {
        if (type == 0) {
          // Point
          std::size_t p = 0;
          gridfile >> p;
          // Make sure the index is not out of range.
          if (p >= nVertices) {
            PrintError(m_className + "::LoadGrid", filename, iLine);
            std::cerr << "    Vertex index out of range.\n";
            return false;
          }
          m_elements[j].vertex[0] = p;
        } else if (type == 1) {
          // Line
          for (size_t k = 0; k < 2; ++k) {
            int p = 0;
            gridfile >> p;
            if (p < 0) p = -p - 1;
            // Make sure the index is not out of range.
            if (p >= (int)nVertices) {
              PrintError(m_className + "::LoadGrid", filename, iLine);
              std::cerr << "    Vertex index out of range.\n";
              return false;
            }
            m_elements[j].vertex[k] = p;
          }
        } else if (type == 2) {
          // Triangle
          int p0 = 0, p1 = 0, p2 = 0;
          gridfile >> p0 >> p1 >> p2;
          // Negative edge index means that the sequence of the two points
          // is supposed to be inverted.
          // The actual index is then given by "-index - 1".
          if (p0 < 0) p0 = -p0 - 1;
          if (p1 < 0) p1 = -p1 - 1;
          if (p2 < 0) p2 = -p2 - 1;
          // Make sure the indices are not out of range.
          if (p0 >= (int)nEdges || p1 >= (int)nEdges || p2 >= (int)nEdges) {
            PrintError(m_className + "::LoadGrid", filename, iLine);
            std::cerr << "    Edge index out of range.\n";
            return false;
          }
          m_elements[j].vertex[0] = edgeP1[p0];
          m_elements[j].vertex[1] = edgeP2[p0];
          if (edgeP1[p1] != m_elements[j].vertex[0] &&
              edgeP1[p1] != m_elements[j].vertex[1]) {
            m_elements[j].vertex[2] = edgeP1[p1];
          } else {
            m_elements[j].vertex[2] = edgeP2[p1];
          }
          // Rearrange vertices such that point 0 is on the left.
          while (m_vertices[m_elements[j].vertex[0]][0] >
                     m_vertices[m_elements[j].vertex[1]][0] ||
                 m_vertices[m_elements[j].vertex[0]][0] >
                     m_vertices[m_elements[j].vertex[2]][0]) {
            const int tmp = m_elements[j].vertex[0];
            m_elements[j].vertex[0] = m_elements[j].vertex[1];
            m_elements[j].vertex[1] = m_elements[j].vertex[2];
            m_elements[j].vertex[2] = tmp;
          }
        } else if (type == 3) {
          // Rectangle
          for (size_t k = 0; k < 4; ++k) {
            int p = 0;
            gridfile >> p;
            // Make sure the index is not out of range.
            if (p >= (int)nEdges || -p - 1 >= (int)nEdges) {
              PrintError(m_className + "::LoadGrid", filename, iLine);
              std::cerr << "    Edge index out of range.\n";
              return false;
            }
            if (p >= 0) {
              m_elements[j].vertex[k] = edgeP1[p];
            } else {
              m_elements[j].vertex[k] = edgeP2[-p - 1];
            }
          }
          // Rearrange vertices such that point 0 is on the left.
          while (m_vertices[m_elements[j].vertex[0]][0] >
                     m_vertices[m_elements[j].vertex[1]][0] ||
                 m_vertices[m_elements[j].vertex[0]][0] >
                     m_vertices[m_elements[j].vertex[2]][0] ||
                 m_vertices[m_elements[j].vertex[0]][0] >
                     m_vertices[m_elements[j].vertex[3]][0]) {
            const int tmp = m_elements[j].vertex[0];
            m_elements[j].vertex[0] = m_elements[j].vertex[1];
            m_elements[j].vertex[1] = m_elements[j].vertex[2];
            m_elements[j].vertex[2] = m_elements[j].vertex[3];
            m_elements[j].vertex[3] = tmp;
          }
        } else {
          // Other element types are not permitted for 2d grids.
          PrintError(m_className + "::LoadGrid", filename, iLine);
          std::cerr << "    Invalid element type (" << type
                    << ") for 2d mesh.\n";
          return false;
        }
      } else if (N == 3) {
        if (type == 0) {
          // Point
          std::size_t p = 0;
          gridfile >> p;
          // Make sure the index is not out of range.
          if (p >= nVertices) {
            PrintError(m_className + "::LoadGrid", filename, iLine);
            std::cerr << "    Vertex index out of range.\n";
            return false;
          }
          m_elements[j].vertex[0] = p;
        } else if (type == 2) {
          // Triangle
          int edge0, edge1, edge2;
          gridfile >> edge0 >> edge1 >> edge2;
          // Get the vertices.
          // Negative edge index means that the sequence of the two points
          // is supposed to be inverted.
          // The actual index is then given by "-index - 1".
          // For our purposes, the orientation does not matter.
          if (edge0 < 0) edge0 = -edge0 - 1;
          if (edge1 < 0) edge1 = -edge1 - 1;
          if (edge2 < 0) edge2 = -edge2 - 1;
          // Make sure the indices are not out of range.
          if (edge0 >= (int)nEdges || edge1 >= (int)nEdges ||
              edge2 >= (int)nEdges) {
            PrintError(m_className + "::LoadGrid", filename, iLine);
            std::cerr << "    Edge index out of range.\n";
            return false;
          }
          m_elements[j].vertex[0] = edgeP1[edge0];
          m_elements[j].vertex[1] = edgeP2[edge0];
          if (edgeP1[edge1] != m_elements[j].vertex[0] &&
              edgeP1[edge1] != m_elements[j].vertex[1]) {
            m_elements[j].vertex[2] = edgeP1[edge1];
          } else {
            m_elements[j].vertex[2] = edgeP2[edge1];
          }
        } else if (type == 5) {
          // Tetrahedron
          // Get the faces.
          // Negative face index means that the sequence of the edges
          // is supposed to be inverted.
          // For our purposes, the orientation does not matter.
          int face0, face1, face2, face3;
          gridfile >> face0 >> face1 >> face2 >> face3;
          if (face0 < 0) face0 = -face0 - 1;
          if (face1 < 0) face1 = -face1 - 1;
          if (face2 < 0) face2 = -face2 - 1;
          if (face3 < 0) face3 = -face3 - 1;
          // Make sure the face indices are not out of range.
          if (face0 >= (int)nFaces || face1 >= (int)nFaces ||
              face2 >= (int)nFaces || face3 >= (int)nFaces) {
            PrintError(m_className + "::LoadGrid", filename, iLine);
            std::cerr << "    Face index out of range.\n";
            return false;
          }
          // Get the edges of the first face.
          int edge0 = faces[face0].edge[0];
          int edge1 = faces[face0].edge[1];
          int edge2 = faces[face0].edge[2];
          if (edge0 < 0) edge0 = -edge0 - 1;
          if (edge1 < 0) edge1 = -edge1 - 1;
          if (edge2 < 0) edge2 = -edge2 - 1;
          // Make sure the edge indices are not out of range.
          if (edge0 >= (int)nEdges || edge1 >= (int)nEdges ||
              edge2 >= (int)nEdges) {
            PrintError(m_className + "::LoadGrid", filename, iLine);
            std::cerr << "    Edge index out of range.\n";
            return false;
          }
          // Get the first three vertices.
          m_elements[j].vertex[0] = edgeP1[edge0];
          m_elements[j].vertex[1] = edgeP2[edge0];
          if (edgeP1[edge1] != m_elements[j].vertex[0] &&
              edgeP1[edge1] != m_elements[j].vertex[1]) {
            m_elements[j].vertex[2] = edgeP1[edge1];
          } else {
            m_elements[j].vertex[2] = edgeP2[edge1];
          }
          // Get the fourth vertex from face 1.
          edge0 = faces[face1].edge[0];
          edge1 = faces[face1].edge[1];
          edge2 = faces[face1].edge[2];
          if (edge0 < 0) edge0 = -edge0 - 1;
          if (edge1 < 0) edge1 = -edge1 - 1;
          if (edge2 < 0) edge2 = -edge2 - 1;
          const auto v0 = m_elements[j].vertex[0];
          const auto v1 = m_elements[j].vertex[1];
          const auto v2 = m_elements[j].vertex[2];
          if (edgeP1[edge0] != v0 && edgeP1[edge0] != v1 &&
              edgeP1[edge0] != v2) {
            m_elements[j].vertex[3] = edgeP1[edge0];
          } else if (edgeP2[edge0] != v0 && edgeP2[edge0] != v1 &&
                     edgeP2[edge0] != v2) {
            m_elements[j].vertex[3] = edgeP2[edge0];
          } else if (edgeP1[edge1] != v0 && edgeP1[edge1] != v1 &&
                     edgeP1[edge1] != v2) {
            m_elements[j].vertex[3] = edgeP1[edge1];
          } else if (edgeP2[edge1] != v0 && edgeP2[edge1] != v1 &&
                     edgeP2[edge1] != v2) {
            m_elements[j].vertex[3] = edgeP2[edge1];
          } else {
            PrintError(m_className + "::LoadGrid", filename, iLine);
            std::cerr << "    Face 1 of element " << j << " is degenerate.\n";
            return false;
          }
        } else {
          // Other element types are not allowed.
          PrintError(m_className + "::LoadGrid", filename, iLine);
          if (type == 0 || type == 1) {
            std::cerr << "    Invalid element type (" << type
                      << ") for 3d mesh.\n";
          } else {
            std::cerr << "    Element type " << type << " is not supported.\n"
                      << "    Remesh with option -t to create only"
                      << " triangles and tetrahedra.\n";
          }
          return false;
        }
      }
      m_elements[j].type = type;
      m_elements[j].region = m_regions.size();
    }
    break;
  }
  if (gridfile.eof()) {
    std::cerr << m_className << "::LoadGrid:\n"
              << "    Could not find section 'Elements' in file\n"
              << "    " << filename << ".\n";
    return false;
  } else if (gridfile.fail()) {
    PrintError(m_className + "::LoadGrid", filename, iLine);
    return false;
  }

  // Assign regions to elements.
  while (std::getline(gridfile, line)) {
    ltrim(line);
    if (line.empty()) continue;
    // Find section 'Region'.
    if (line.substr(0, 6) != "Region") continue;
    // Get region name (given in brackets).
    if (!ExtractFromBrackets(line)) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Could not read region name.\n";
      return false;
    }
    std::istringstream data(line);
    ltrim(line);
    rtrim(line);
    const std::string name = line;
    // std::string name;
    // data >> name;
    // data.clear();
    const size_t index = FindRegion(name);
    if (index >= m_regions.size()) {
      // Specified region name is not in the list.
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Unknown region " << name << ".\n";
      continue;
    }
    std::getline(gridfile, line);
    std::getline(gridfile, line);
    if (!ExtractFromBrackets(line)) {
      std::cerr << m_className << "::LoadGrid:\n"
                << "    Could not read number of elements in region " << name
                << ".\n";
      return false;
    }
    int nElementsRegion;
    data.str(line);
    data.clear();
    data >> nElementsRegion;
    for (int j = 0; j < nElementsRegion; ++j) {
      size_t iElement = 0;
      gridfile >> iElement;
      if (iElement >= m_elements.size()) {
        std::cerr << m_className << "::LoadGrid:\n"
                  << "    Error reading element indices for region " << name
                  << ".\n";
        return false;
      }
      m_elements[iElement].region = index;
    }
  }
  if (gridfile.fail() && !gridfile.eof()) {
    std::cerr << m_className << "::LoadGrid:\n"
              << "    Error reading file " << filename << ".\n";
    return false;
  }
  return true;
}

template <std::size_t N>
bool ComponentTcadBase<N>::LoadSilvaco(const std::string& filename) {
  if (N != 2){
    std::cerr << m_className << "::LoadSilvaco:\n"
              << "    Only 2D Silvaco meshes are supported.\n";
    return false;
  }
  // Electrostatic Potential
  constexpr int kCodePotential = 100; 
  // ElectricField_x
  constexpr int kCodeEx = 120;
  // ElectricField_y
  constexpr int kCodeEy = 121; 
  // Unit conversion from microns to centimeters       
  constexpr double kUmToCm = 1.0e-4;

  std::ifstream infile(filename);
  if (!infile) {
    std::cerr << m_className << "::LoadSilvaco:\n"
              << "    Could not open file " << filename << ".\n";
    return false;
  }
  Cleanup();
  // Silvaco c-ids are 1-based; n-ids are 0-based
  std::map<std::size_t, std::size_t> coordIndex;
  std::vector<int> codes;
  int colPot = -1, colEx = -1, colEy = -1;
  // Per-node {potential, Ex, Ey}, keyed by silvaco node id (0-based).
  std::map<std::size_t, std::array<double, 3>> nodeData;
  // region_id -> material_code (from 'r' records).
  std::map<std::size_t, int> regionMaterial;

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;
    const char tag = line[0];
    // vertex
    if (tag == 'c' && line.size() > 1 && line[1] == ' ') {
      std::istringstream ss(line.substr(1));
      std::size_t cid;
      double x, y, z = 0.;
      ss >> cid >> x >> y >> z;
      if (ss.fail()) continue;
      std::array<double, N> v;
      v[0] = x * kUmToCm;
      v[1] = y * kUmToCm;
      coordIndex[cid] = m_vertices.size();
      m_vertices.push_back(v);
      continue;
    }
    // region
    if (tag == 'r' && line.size() > 1 && line[1] == ' '){
      std::istringstream ss(line.substr(1));
      std::size_t rid;
      int mat;
      ss >> rid >> mat;
      if (!ss.fail()) regionMaterial[rid] = mat;
      continue;
    }
    // solution spec for t
    if (tag == 't' && line.size() > 1 && line[1] == ' ') {
      std::istringstream ss(line.substr(1));
      std::size_t tid, region, a, b, c;
      ss >> tid >> region >> a >> b >> c;
      if (ss.fail()) continue;
      Element element;
      element.type = 2;
      element.region = region;
      element.vertex[0] = a;
      element.vertex[1] = b;
      element.vertex[2] = c;
      element.vertex[3] = 0;
      m_elements.push_back(element);
      continue;
    }
    // solution spec for s
    if (tag == 's' && line.size() > 1 && line[1] == ' ') {
      std::istringstream ss(line.substr(1));
      std::size_t ncode;
      ss >> ncode;
      codes.clear();
      for (std::size_t i = 0; i < ncode; ++i) {
        int code;
        ss >> code;
        codes.push_back(code);
      }
      for (std::size_t i = 0; i < codes.size(); ++i) {
        if (codes[i] == kCodePotential) colPot = static_cast<int>(i);
        else if (codes[i] == kCodeEx) colEx = static_cast<int>(i);
        else if (codes[i] == kCodeEy) colEy = static_cast<int>(i);
      }
      continue;
    }
    if (tag == 'n' && line.size() > 1 && line[1] == ' ' && !codes.empty()) {
      std::istringstream ss(line.substr(1));
      std::size_t nid;
      int fieldA;
      ss >> nid >> fieldA;
      if (ss.fail()) continue;
      if (fieldA == 0) continue;
      std::vector<double> vals;
      vals.reserve(codes.size());
      double val;
      while (ss >> val) vals.push_back(val);
      if (vals.size() < codes.size()) continue;
      std::array<double, 3> d = {0., 0., 0.};
      if (colPot >= 0) d[0] = vals[colPot];
      if (colEx >= 0) d[1] = vals[colEx];
      if (colEy >= 0) d[2] = vals[colEy];
      nodeData[nid] = d;
      continue;
    }
  }

  if (m_vertices.empty() || m_elements.empty()) {
    std::cerr << m_className << "::LoadSilvaco:\n"
              << "    No vertices or elements read from " << filename << ".\n";
    return false;
  }

  // Pass 2: resolve element vertex ids -> m_vertices indices.
  for (auto& element : m_elements) {
    for (std::size_t k = 0; k < 3; ++k) {
      const auto it = coordIndex.find(element.vertex[k]);
      if (it == coordIndex.end()) {
        std::cerr << m_className << "::LoadSilvaco:\n"
                  << "    Element references unknown vertex id "
                  << element.vertex[k] << ".\n";
        return false;
      }
      element.vertex[k] = it->second;
    }
  }

  // Size to the max region id
  std::size_t nRegions = regionMaterial.size();
  for (const auto& element : m_elements) {
    nRegions = std::max(nRegions, element.region + 1);
  }
  m_regions.resize(nRegions);
  for (std::size_t j = 0; j < nRegions; ++j) {
    m_regions[j].name = "region_" + std::to_string(j);
    const auto it = regionMaterial.find(j);
    m_regions[j].material =
        it != regionMaterial.end() ? std::to_string(it->second) : "";
    m_regions[j].drift = true;
    m_regions[j].medium = nullptr;
  }

  // Potential & field per vertex (coord_id = node_id + 1).
  const std::size_t nV = m_vertices.size();
  m_epot.assign(nV, 0.);
  m_efield.assign(nV, {});
  const bool haveField = (colEx >= 0 && colEy >= 0);
  const bool havePot = (colPot >= 0);
  for (const auto& kv : nodeData) {
    const std::size_t nid = kv.first;
    const auto cit = coordIndex.find(nid + 1);
    if (cit == coordIndex.end()) continue;
    const std::size_t idx = cit->second;
    if (havePot) m_epot[idx] = kv.second[0];
    if (haveField) {
      m_efield[idx][0] = kv.second[1];   // Ex
      m_efield[idx][1] = kv.second[2];   // Ey
    }
  }
  if (!haveField) m_efield.clear();
  if (!havePot) m_epot.clear();

  if (m_debug) {
    std::cout << m_className << "::LoadSilvaco:\n"
              << "    Read " << m_vertices.size() << " vertices, "
              << m_elements.size() << " elements, " << m_regions.size()
              << " regions from " << filename << ".\n";
  }
  return true;
}

template <std::size_t N>
bool ComponentTcadBase<N>::InitialiseSilvaco(const std::string& filename) {
  m_ready = false;

  if (!LoadSilvaco(filename)) {
    Cleanup();
    return false;
  }

  // Element bounding boxes + min edge length
  for (std::size_t i = 0; i < N; ++i) {
    m_bbMin[i] = m_vertices[m_elements[0].vertex[0]][i];
    m_bbMax[i] = m_vertices[m_elements[0].vertex[0]][i];
  }
  const std::size_t nElements = m_elements.size();
  for (std::size_t i = 0; i < nElements; ++i) {
    Element& element = m_elements[i];
    std::array<double, N> xmin = m_vertices[element.vertex[0]];
    std::array<double, N> xmax = m_vertices[element.vertex[0]];
    const auto nVe = ElementVertices(element);
    for (std::size_t j = 0; j < nVe; ++j) {
      const auto& v = m_vertices[element.vertex[j]];
      for (std::size_t k = 0; k < N; ++k) {
        xmin[k] = std::min(xmin[k], v[k]);
        xmax[k] = std::max(xmax[k], v[k]);
        m_bbMin[k] = std::min(m_bbMin[k], v[k]);
        m_bbMax[k] = std::max(m_bbMax[k], v[k]);
      }
    }
    double distMin = -1.;
    for (std::size_t j = 0; j < nVe; ++j) {
      for (std::size_t k = j + 1; k < nVe; ++k) {
        const auto& vj = m_vertices[element.vertex[j]];
        const auto& vk = m_vertices[element.vertex[k]];
        double dd = 0.;
        for (std::size_t m = 0; m < N; ++m) {
          dd += (vj[m] - vk[m]) * (vj[m] - vk[m]);
        }
        dd = std::sqrt(dd);
        if (distMin < 0. || dd < distMin) distMin = dd;
      }
    }
    element.length = distMin;
    const double tol = 1.e-6 * (distMin > 0. ? distMin : 1.);
    for (std::size_t k = 0; k < N; ++k) {
      element.bbMin[k] = static_cast<float>(xmin[k] - tol);
      element.bbMax[k] = static_cast<float>(xmax[k] + tol);
    }
  }

  std::cout << m_className << "::InitialiseSilvaco:\n"
            << "    Number of regions: " << m_regions.size() << "\n"
            << "    Number of vertices: " << m_vertices.size() << "\n"
            << "    Number of elements: " << m_elements.size() << "\n";

  if (!m_epot.empty()) {
    m_pMin = m_pMax = m_epot[0];
    for (const double p : m_epot) {
      m_pMin = std::min(m_pMin, p);
      m_pMax = std::max(m_pMax, p);
    }
  }

  FillTree();
  m_ready = true;
  UpdatePeriodicity();
  std::cout << m_className << "::InitialiseSilvaco: Initialisation finished.\n";
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::LoadData(const std::string& filename) {
  std::ifstream datafile(filename);
  if (!datafile) {
    std::cerr << m_className << "::LoadData:\n"
              << "    Could not open file " << filename << ".\n";
    return false;
  }
  const size_t nVertices = m_vertices.size();
  std::vector<std::size_t> fillCount(nVertices, 0);

  m_defects.clear();
  std::array<double, N> zeros;
  zeros.fill(0.);
  // Read the file line by line.
  std::string line;
  while (std::getline(datafile, line)) {
    // Strip white space from the beginning of the line.
    ltrim(line);
    // Skip empty lines.
    if (line.empty()) continue;
    if (startsWith(line, "functions")) SetupDefects(line);
    // Find data section.
    if (line.substr(0, 9) != "function ") continue;
    // Read type of data set.
    auto pEq = line.find('=');
    if (pEq == std::string::npos) {
      // No "=" found.
      std::cerr << m_className << "::LoadData:\n"
                << "    Error reading file " << filename << ".\n"
                << "    Line:\n    " << line << "\n";
      return false;
    }
    line = line.substr(pEq + 1);
    std::string dataset;
    std::istringstream data(line);
    data >> dataset;
    data.clear();
    std::getline(datafile, line);
    pEq = line.find('=');
    if (pEq == std::string::npos) {
      // No "=" found.
      std::cerr << m_className << "::LoadData:\n"
                << "    Error reading file " << filename << ".\n"
                << "    Line:\n    " << line << "\n";
      return false;
    }
    line = line.substr(pEq + 1);
    std::string dstype;
    data.str(line);
    data.clear();
    data >> dstype;
    if (m_debug && dataset != "[") {
      std::cout << m_className << "::LoadData: Found " << dstype << " dataset "
                << dataset << ".\n";
    }
    if (dstype == "scalar" &&
        (dataset == "ElectricField" || dataset == "eDriftVelocity" ||
         dataset == "hDriftVelocity")) {
      if (m_debug) std::cout << "    Skipping this dataset.\n";
      continue;
    }
    if (dataset == "ElectrostaticPotential") {
      if (m_epot.empty()) m_epot.assign(nVertices, 0.);
      if (!ReadDataset(datafile, dataset)) {
        m_epot.clear();
        return false;
      }
    } else if (dataset == "ElectricField") {
      if (m_efield.empty()) m_efield.assign(nVertices, zeros);
      if (!ReadDataset(datafile, dataset)) {
        m_efield.clear();
        return false;
      }
    } else if (dataset == "eDriftVelocity") {
      if (m_eVelocity.empty()) m_eVelocity.assign(nVertices, zeros);
      if (!ReadDataset(datafile, dataset)) {
        m_eVelocity.clear();
        return false;
      }
    } else if (dataset == "hDriftVelocity") {
      if (m_hVelocity.empty()) m_hVelocity.assign(nVertices, zeros);
      if (!ReadDataset(datafile, dataset)) {
        m_hVelocity.clear();
        return false;
      }
    } else if (dataset == "eMobility") {
      if (m_eMobility.empty()) m_eMobility.assign(nVertices, 0.);
      if (!ReadDataset(datafile, dataset)) {
        m_eMobility.clear();
        return false;
      }
    } else if (dataset == "hMobility") {
      if (m_hMobility.empty()) m_hMobility.assign(nVertices, 0.);
      if (!ReadDataset(datafile, dataset)) {
        m_hMobility.clear();
        return false;
      }
    } else if (dataset == "eAlphaAvalanche") {
      if (m_eAlpha.empty()) m_eAlpha.assign(nVertices, 0.);
      if (!ReadDataset(datafile, dataset)) {
        m_eAlpha.clear();
        return false;
      }
    } else if (dataset == "hAlphaAvalanche") {
      if (m_hAlpha.empty()) m_hAlpha.assign(nVertices, 0.);
      if (!ReadDataset(datafile, dataset)) {
        m_hAlpha.clear();
        return false;
      }
    } else if (dataset == "eLifetime") {
      if (m_eLifetime.empty()) m_eLifetime.assign(nVertices, 0.);
      if (!ReadDataset(datafile, dataset)) {
        m_eLifetime.clear();
        return false;
      }
    } else if (dataset == "hLifetime") {
      if (m_hLifetime.empty()) m_hLifetime.assign(nVertices, 0.);
      if (!ReadDataset(datafile, dataset)) {
        m_hLifetime.clear();
        return false;
      }
    } else if (dataset.substr(0, 14) == "TrapOccupation") {
      if (m_defects.empty()) {
        std::cerr << m_className << "::LoadData:\n"
                  << "    Unexpected data set: " << dataset << ".\n";
      } else {
        if (m_defectOcc.empty()) {
          std::vector<float> zerosOcc(m_defects.size(), 0.);
          m_defectOcc.assign(nVertices, zerosOcc);
        }
        if (!ReadDataset(datafile, dataset)) {
          m_defectOcc.clear();
          return false;
        }
      }
    }
  }
  if (datafile.fail() && !datafile.eof()) {
    std::cerr << m_className << "::LoadData:\n"
              << "    Error reading file " << filename << "\n";
    return false;
  }
  if (m_useTrapOccMap || m_useLifetimeMap) UpdateAttachment();
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::SetupDefects(const std::string& line) {
  std::string functions = line;
  if (!ExtractFromSquareBrackets(functions)) {
    return false;
  }
  for (auto& token : tokenize(functions)) {
    if (!startsWith(token, "TrapOccupation")) continue;
    // Get the index.
    const int nDefects = m_defects.size();
    const int index = GetTrapIndex(token);
    if (index < 0) {
      std::cerr << m_className << "::SetupDefects:\n"
                << "    Cannot extract index from " << token << ".\n";
      return false;
    } else if (index < nDefects) {
      // Defect already created.
      continue;
    } else if (index != nDefects) {
      std::cerr << m_className << "::SetupDefects:\n"
                << "    Unexpected index (" << index << ").\n";
    }
    Defect defect;
    defect.xsece = -1.;
    defect.xsech = -1.;
    defect.conc = -1.;
    // Determine the defect type (donor or acceptor).
    ExtractFromBrackets(token);
    if (startsWith(token, "Do")) {
      defect.type = DefectType::Donor;
    } else if (startsWith(token, "Ac")) {
      defect.type = DefectType::Acceptor;
    } else {
      std::cerr << m_className << "::SetupDefects:\n"
                << "    Unexpected defect type (" << token << ").\n";
      continue;
    }
    m_defects.push_back(std::move(defect));
  }
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::ReadDataset(std::ifstream& datafile,
                                       const std::string& dataset) {
  if (!datafile.is_open()) return false;
  enum DataSet {
    ElectrostaticPotential,
    EField,
    eDriftVelocity,
    hDriftVelocity,
    eMobility,
    hMobility,
    eAlpha,
    hAlpha,
    eLifetime,
    hLifetime,
    TrapOccupation,
    Unknown
  };
  int trapIndex = 0;
  DataSet ds = Unknown;
  if (dataset == "ElectrostaticPotential") {
    ds = ElectrostaticPotential;
  } else if (dataset == "ElectricField") {
    ds = EField;
  } else if (dataset == "eDriftVelocity") {
    ds = eDriftVelocity;
  } else if (dataset == "hDriftVelocity") {
    ds = hDriftVelocity;
  } else if (dataset == "eMobility") {
    ds = eMobility;
  } else if (dataset == "hMobility") {
    ds = hMobility;
  } else if (dataset == "eAlphaAvalanche") {
    ds = eAlpha;
  } else if (dataset == "hAlphaAvalanche") {
    ds = hAlpha;
  } else if (dataset == "eLifetime") {
    ds = eLifetime;
  } else if (dataset == "hLifetime") {
    ds = hLifetime;
  } else if (dataset.substr(0, 14) == "TrapOccupation") {
    ds = TrapOccupation;
    trapIndex = GetTrapIndex(dataset);
    if (trapIndex < 0 || trapIndex >= (int)m_defects.size()) {
      std::cerr << m_className << "::ReadDataset:\n"
                << "    Unexpected trap index " << trapIndex << ".\n";
      return false;
    }
  } else {
    std::cerr << m_className << "::ReadDataset:\n"
              << "    Unexpected dataset " << dataset << ".\n";
    return false;
  }
  bool isVector = false;
  if (ds == EField || ds == eDriftVelocity || ds == hDriftVelocity) {
    isVector = true;
  }
  std::string line;
  std::getline(datafile, line);
  std::getline(datafile, line);
  std::getline(datafile, line);
  // Get the region name (given in brackets).
  if (!ExtractFromSquareBrackets(line)) {
    std::cerr << m_className << "::ReadDataset:\n"
              << "    Cannot extract region name.\n"
              << "    Line:\n    " << line << "\n";
    return false;
  }
  std::string name;
  std::istringstream data(line);
  data >> name;
  data.clear();
  // Check if the region name matches one from the mesh file.
  const size_t index = FindRegion(name);
  if (index >= m_regions.size()) {
    std::cerr << m_className << "::ReadDataset:\n"
              << "    Unknown region " << name << ".\n";
    return false;
  }
  if (m_debug) {
    std::cout << m_className << "::ReadDataset:\n"
              << "    Reading dataset " << dataset << " for region " << name
              << ".\n";
  }
  // Get the number of values.
  std::getline(datafile, line);
  if (!ExtractFromBrackets(line)) {
    std::cerr << m_className << "::ReadDataset:\n"
              << "    Cannot extract number of values to be read.\n"
              << "    Line:\n    " << line << "\n";
    return false;
  }
  int nValues;
  data.str(line);
  data.clear();
  data >> nValues;
  if (isVector) nValues /= N;
  if (m_debug) std::cout << "    Expecting " << nValues << " values.\n";
  // Mark the vertices belonging to this region.
  const size_t nVertices = m_vertices.size();
  std::vector<bool> isInRegion(nVertices, false);
  size_t nVerticesInRegion = 0;
  size_t nElementsInRegion = 0;
  for (const auto& element : m_elements) {
    if (element.region != index) continue;
    ++nElementsInRegion;
    const std::size_t nV = ElementVertices(element);
    for (std::size_t k = 0; k < nV; ++k) {
      if (isInRegion[element.vertex[k]]) continue;
      isInRegion[element.vertex[k]] = true;
      ++nVerticesInRegion;
    }
  }
  if (m_debug) {
    std::cout << "    Region has " << nElementsInRegion << " elements and "
              << nVerticesInRegion << " vertices.\n";
  }
  std::size_t ivertex = 0;
  for (int j = 0; j < nValues; ++j) {
    // Read the next value.
    std::array<long double, N> val;
    if (isVector) {
      for (size_t k = 0; k < N; ++k) datafile >> val[k];
    } else {
      datafile >> val[0];
    }
    // Find the next vertex belonging to the region.
    while (ivertex < nVertices) {
      if (isInRegion[ivertex]) break;
      ++ivertex;
    }
    // Check if there is a mismatch between the number of vertices
    // and the number of potential values.
    if (ivertex >= nVertices) {
      std::cerr << m_className << "::ReadDataset:\n"
                << "    Dataset " << dataset << " has more values than "
                << "there are vertices in region " << name << "\n";
      return false;
    }

    switch (ds) {
      case ElectrostaticPotential:
        m_epot[ivertex] = val[0];
        break;
      case EField:
        for (size_t k = 0; k < N; ++k) m_efield[ivertex][k] = val[k];
        break;
      case eDriftVelocity:
        // Scale from cm/s to cm/ns.
        for (size_t k = 0; k < N; ++k) {
          m_eVelocity[ivertex][k] = val[k] * 1.e-9;
        }
        break;
      case hDriftVelocity:
        // Scale from cm/s to cm/ns.
        for (size_t k = 0; k < N; ++k) {
          m_hVelocity[ivertex][k] = val[k] * 1.e-9;
        }
        break;
      case eMobility:
        // Convert from cm2 / (V s) to cm2 / (V ns).
        m_eMobility[ivertex] = val[0] * 1.e-9;
        break;
      case hMobility:
        // Convert from cm2 / (V s) to cm2 / (V ns).
        m_hMobility[ivertex] = val[0] * 1.e-9;
        break;
      case eAlpha:
        m_eAlpha[ivertex] = val[0];
        break;
      case hAlpha:
        m_hAlpha[ivertex] = val[0];
        break;
      case eLifetime:
        // Convert from s to ns.
        m_eLifetime[ivertex] = val[0] * 1.e9;
        break;
      case hLifetime:
        // Convert from s to ns.
        m_hLifetime[ivertex] = val[0] * 1.e9;
        break;
      case TrapOccupation:
        m_defectOcc[ivertex][trapIndex] = val[0];
        break;
      default:
        std::cerr << m_className << "::ReadDataset:\n"
                  << "    Unexpected dataset (" << ds << "). Program bug!\n";
        return false;
    }
    ++ivertex;
  }
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::LoadWeightingField(
    const std::string& filename, std::vector<std::array<double, N> >& wf,
    std::vector<double>& wp) {
  std::ifstream datafile(filename, std::ios::in);
  if (!datafile) {
    std::cerr << m_className << "::LoadWeightingField:\n"
              << "    Could not open file " << filename << ".\n";
    return false;
  }
  const size_t nVertices = m_vertices.size();
  std::array<double, N> zeros;
  zeros.fill(0.);
  bool ok = true;
  // Read the file line by line.
  std::string line;
  while (std::getline(datafile, line)) {
    // Strip white space from the beginning of the line.
    ltrim(line);
    if (line.empty()) continue;
    // Find data section.
    if (line.substr(0, 9) != "function ") continue;
    // Read type of data set.
    auto pEq = line.find('=');
    if (pEq == std::string::npos) {
      // No "=" found.
      std::cerr << m_className << "::LoadWeightingField:\n"
                << "    Error reading file " << filename << ".\n"
                << "    Line:\n    " << line << "\n";
      return false;
    }
    line = line.substr(pEq + 1);
    std::string dataset;
    std::istringstream data(line);
    data >> dataset;
    data.clear();
    if (dataset != "ElectrostaticPotential" && dataset != "ElectricField") {
      continue;
    }
    std::getline(datafile, line);
    pEq = line.find('=');
    if (pEq == std::string::npos) {
      // No "=" found.
      std::cerr << m_className << "::LoadWeightingField:\n"
                << "    Error reading file " << filename << ".\n"
                << "    Line:\n    " << line << "\n";
      return false;
    }
    line = line.substr(pEq + 1);
    std::string dstype;
    data.str(line);
    data.clear();
    data >> dstype;
    if (dstype == "scalar" && dataset == "ElectricField") {
      if (m_debug) {
        std::cout << m_className << "::LoadWeightingField:\n"
                  << "    Skipping scalar dataset " << dataset << ".\n";
      }
      continue;
    }
    bool field = false;
    if (dataset == "ElectricField") {
      if (wf.empty()) wf.assign(nVertices, zeros);
      field = true;
    } else {
      if (wp.empty()) wp.assign(nVertices, 0.);
    }
    std::getline(datafile, line);
    std::getline(datafile, line);
    std::getline(datafile, line);
    // Get the region name (given in brackets).
    if (!ExtractFromSquareBrackets(line)) {
      std::cerr << m_className << "::LoadWeightingField:\n"
                << "    Cannot extract region name.\n"
                << "    Line:\n    " << line << "\n";
      ok = false;
      break;
    }
    ltrim(line);
    rtrim(line);
    const std::string name = line;
    // Check if the region name matches one from the mesh file.
    const auto index = FindRegion(name);
    if (index >= m_regions.size()) {
      std::cerr << m_className << "::LoadWeightingField:\n"
                << "    Unknown region " << name << ".\n";
      ok = false;
      break;
    }
    // Get the number of values.
    std::getline(datafile, line);
    if (!ExtractFromBrackets(line)) {
      std::cerr << m_className << "::LoadWeightingField:\n"
                << "    Cannot extract number of values to be read.\n"
                << "    Line:\n    " << line << "\n";
      ok = false;
      break;
    }
    int nValues;
    data.str(line);
    data.clear();
    data >> nValues;
    if (field) nValues /= N;
    // Mark the vertices belonging to this region.
    std::vector<bool> isInRegion(nVertices, false);
    const size_t nElements = m_elements.size();
    for (size_t j = 0; j < nElements; ++j) {
      if (m_elements[j].region != index) continue;
      const std::size_t nV = ElementVertices(m_elements[j]);
      for (std::size_t k = 0; k < nV; ++k) {
        isInRegion[m_elements[j].vertex[k]] = true;
      }
    }
    std::size_t ivertex = 0;
    for (int j = 0; j < nValues; ++j) {
      // Read the next value.
      std::array<double, N> val;
      if (field) {
        for (size_t k = 0; k < N; ++k) datafile >> val[k];
      } else {
        datafile >> val[0];
      }
      // Find the next vertex belonging to the region.
      while (ivertex < nVertices) {
        if (isInRegion[ivertex]) break;
        ++ivertex;
      }
      // Check if there is a mismatch between the number of vertices
      // and the number of values.
      if (ivertex >= nVertices) {
        std::cerr << m_className << "::LoadWeightingField:\n"
                  << "    Dataset " << dataset
                  << " has more values than vertices in region " << name
                  << "\n";
        ok = false;
        break;
      }
      if (field) {
        wf[ivertex] = val;
      } else {
        wp[ivertex] = val[0];
      }
      ++ivertex;
    }
  }

  if (!ok || (datafile.fail() && !datafile.eof())) {
    std::cerr << m_className << "::LoadWeightingField:\n"
              << "    Error reading file " << filename << "\n";
    return false;
  }
  return true;
}

template <size_t N>
void ComponentTcadBase<N>::PrintRegions() const {
  if (m_regions.empty()) {
    std::cerr << m_className << "::PrintRegions:\n"
              << "    No regions are currently defined.\n";
    return;
  }

  const size_t nRegions = m_regions.size();
  std::cout << m_className << "::PrintRegions:\n"
            << "    Currently " << nRegions << " regions are defined.\n"
            << " Index   Name               Material            Medium\n";
  for (size_t i = 0; i < nRegions; ++i) {
    std::cout << std::setw(8) << std::right << i << " " << std::setw(20)
              << std::left << m_regions[i].name << " " << std::setw(18)
              << std::left << m_regions[i].material << " ";
    if (!m_regions[i].medium) {
      std::cout << std::setw(18) << "none";
    } else {
      std::cout << std::setw(18) << m_regions[i].medium->GetName();
    }
    if (m_regions[i].drift) {
      std::cout << " (active)\n";
    } else {
      std::cout << "\n";
    }
  }
}

template <size_t N>
void ComponentTcadBase<N>::GetRegion(const std::size_t i, std::string& name,
                                     bool& active) const {
  if (i >= m_regions.size()) {
    std::cerr << m_className << "::GetRegion: Index out of range.\n";
    return;
  }
  name = m_regions[i].name;
  active = m_regions[i].drift;
}

template <size_t N>
void ComponentTcadBase<N>::SetDriftRegion(const size_t i) {
  if (i >= m_regions.size()) {
    std::cerr << m_className << "::SetDriftRegion: Index out of range.\n";
    return;
  }
  m_regions[i].drift = true;
}

template <size_t N>
void ComponentTcadBase<N>::UnsetDriftRegion(const size_t i) {
  if (i >= m_regions.size()) {
    std::cerr << m_className << "::UnsetDriftRegion: Index out of range.\n";
    return;
  }
  m_regions[i].drift = false;
}

template <std::size_t N>
void ComponentTcadBase<N>::SetMedium(const std::size_t index, Medium* medium) {
  if (index >= m_regions.size())
    throw Exception("::SetMedium: Index out of range");
  if (!medium) throw Exception("::SetMedium: Null pointer");
  m_regions[index].medium = medium;
}

template <std::size_t N>
void ComponentTcadBase<N>::SetMedium(const std::string& material,
                                     Medium* medium) {
  if (!medium) throw Exception("::SetMedium: Null pointer");
  size_t nMatch = 0;
  const auto nRegions = m_regions.size();
  for (size_t i = 0; i < nRegions; ++i) {
    if (material != m_regions[i].material) continue;
    m_regions[i].medium = medium;
    std::cout << m_className << "::SetMedium: Associating region " << i << " ("
              << m_regions[i].name << ") with " << medium->GetName() << ".\n";
    ++nMatch;
  }
  if (nMatch == 0) {
    std::cerr << m_className << "::SetMedium: Found no region with material "
              << material << ".\n";
  }
}

template <size_t N>
bool ComponentTcadBase<N>::SetDefect(const size_t i, const double eXsec,
                                     const double hXsec, const double conc) {
  if (i >= m_defects.size()) {
    std::cerr << m_className << "::SetDefect: Index out of range.\n";
    return false;
  }
  m_defects[i].xsece = eXsec;
  m_defects[i].xsech = hXsec;
  m_defects[i].conc = conc;

  UpdateAttachment();
  return true;
}

template <size_t N>
void ComponentTcadBase<N>::PrintDefects() const {
  if (m_defects.empty()) {
    std::cerr << m_className << "::PrintDefects:\n"
              << "    There are no trap states in the present map.\n";
    return;
  }
  std::cout << m_className << "::PrintDefects:\n";
  std::cout << "   Index     Type         Cross-section [cm2]      "
               "Concentration [cm-3]\n";
  std::cout << "                          Electrons       Holes\n";
  for (size_t i = 0; i < m_defects.size(); ++i) {
    std::string stype = "Acceptor";
    if (m_defects[i].type == DefectType::Donor) stype = "Donor   ";
    if (m_defects[i].xsece < 0. || m_defects[i].xsech < 0. ||
        m_defects[i].conc < 0.) {
      std::string sundef(15, '-');
      std::printf("%5zu    %s  %s %s %s\n", i, stype.c_str(), sundef.c_str(),
                  sundef.c_str(), sundef.c_str());
    } else {
      std::printf("%5zu    %s  %15.5e %15.5e %15.5e\n", i, stype.c_str(),
                  m_defects[i].xsece, m_defects[i].xsech, m_defects[i].conc);
    }
  }
}

template <size_t N>
bool ComponentTcadBase<N>::ElectronAttachment(const double x, const double y,
                                              const double z, double& eta) {
  Interpolate(x, y, z, m_eEta, eta);
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::HoleAttachment(const double x, const double y,
                                          const double z, double& eta) {
  Interpolate(x, y, z, m_hEta, eta);
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::ElectronTownsend(const double x, const double y,
                                            const double z, double& alpha) {
  Interpolate(x, y, z, m_eAlpha, alpha);
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::HoleTownsend(const double x, const double y,
                                        const double z, double& alpha) {
  Interpolate(x, y, z, m_hAlpha, alpha);
  return true;
}

template <size_t N>
bool ComponentTcadBase<N>::ElectronVelocity(const double x, const double y,
                                            const double z, double& vx,
                                            double& vy, double& vz) {
  return Interpolate(x, y, z, m_eVelocity, vx, vy, vz);
}

template <size_t N>
bool ComponentTcadBase<N>::HoleVelocity(const double x, const double y,
                                        const double z, double& vx, double& vy,
                                        double& vz) {
  return Interpolate(x, y, z, m_hVelocity, vx, vy, vz);
}

template <size_t N>
double ComponentTcadBase<N>::ElectronLifetime(const double x, const double y,
                                              const double z) {
  double tau = 0.;
  Interpolate(x, y, z, m_eLifetime, tau);
  return tau;
}

template <size_t N>
double ComponentTcadBase<N>::HoleLifetime(const double x, const double y,
                                          const double z) {
  double tau = 0.;
  Interpolate(x, y, z, m_hLifetime, tau);
  return tau;
}

template <size_t N>
bool ComponentTcadBase<N>::ElectronMobility(const double x, const double y,
                                            const double z, double& mu) {
  return Interpolate(x, y, z, m_eMobility, mu);
}

template <size_t N>
bool ComponentTcadBase<N>::HoleMobility(const double x, const double y,
                                        const double z, double& mu) {
  return Interpolate(x, y, z, m_hMobility, mu);
}

template <size_t N>
void ComponentTcadBase<N>::UpdatePeriodicity() {
  if (!m_ready) {
    std::cerr << m_className << "::UpdatePeriodicity:\n"
              << "    Field map not available.\n";
    return;
  }

  // Check for conflicts.
  for (size_t i = 0; i < 3; ++i) {
    const Symmetry sym = m_symmetries.GetSymmetries(i);
    if (sym.Has(Symmetry::Type::Periodic) && sym.Has(Symmetry::Type::Mirror)) {
      std::cerr << m_className << "::UpdatePeriodicity:\n"
                << "    Both simple and mirror periodicity requested. Reset.\n";
      m_symmetries.Deactivate(Symmetry::Type::Periodic);
      m_symmetries.Deactivate(Symmetry::Type::Mirror);
    }

    if (sym.Has(Symmetry::Type::Periodic) &&
        sym.Has(Symmetry::Type::MirrorFinite)) {
      std::cerr
          << m_className << "::UpdatePeriodicity:\n"
          << "    Simple periodicity and finite mirror requested. Reset.\n";
      m_symmetries.Deactivate(Symmetry::Type::Periodic);
      m_symmetries.Deactivate(Symmetry::Type::MirrorFinite);
    }

    if (sym.Has(Symmetry::Type::Mirror) &&
        sym.Has(Symmetry::Type::MirrorFinite)) {
      std::cerr << m_className << "::UpdatePeriodicity:\n"
                << "    Infinite mirror and finite mirror requested. Reset.\n";
      m_symmetries.Deactivate(Symmetry::Type::Mirror);
      m_symmetries.Deactivate(Symmetry::Type::MirrorFinite);
    }
  }
  if (m_symmetries.Has(Symmetry::Type::Axial)) {
    std::cerr << m_className << "::UpdatePeriodicity:\n"
              << "    Axial symmetry is not supported. Reset.\n";
    m_symmetries.Deactivate(Symmetry::Type::Axial);
  }
  if (m_symmetries.Has(Symmetry::Type::Rotation)) {
    std::cerr << m_className << "::UpdatePeriodicity:\n"
              << "    Rotation symmetry is not supported. Reset.\n";
    m_symmetries.Deactivate(Symmetry::Type::TriangleXZ);
  }

  // Triangle symmetry
  if constexpr (N == 2) {
    // No xz or yz planes in 2D.
    if (m_symmetries.Has(Symmetry::Type::TriangleXZ) ||
        m_symmetries.Has(Symmetry::Type::TriangleYZ)) {
      std::cerr << m_className << "::UpdatePeriodicity2d:\n"
                << "    Triangle periodicity does not allow\n"
                << "    for xz or yz planes; reset.\n";
      m_symmetries.Deactivate(Symmetry::Type::Triangle);
      m_symmetries.Deactivate(Symmetry::Type::Mirror);
      m_symmetries.Deactivate(Symmetry::Type::MirrorFinite);
      m_triangleSymmetricOct = 0;
    }
  }

  if constexpr (N == 3) {
    const int triSymmN = m_symmetries.NumberOf(Symmetry::Type::Triangle);
    if (triSymmN == 1) {
      if (m_triangleSymmetricOct < 1 || m_triangleSymmetricOct > 8) {
        std::cerr << m_className << "::UpdatePeriodicity:\n"
                  << "    For triangle symmetry, octant must be in the range "
                     "1–8; reset.\n";
        m_symmetries.Deactivate(Symmetry::Type::Triangle);
        m_symmetries.Deactivate(Symmetry::Type::Mirror);
        m_symmetries.Deactivate(Symmetry::Type::MirrorFinite);
        m_triangleSymmetricOct = 0;
      } else {
        m_outsideCone = (m_triangleOctRules[0] == m_triangleSymmetricOct) ||
                        (m_triangleOctRules[1] == m_triangleSymmetricOct) ||
                        (m_triangleOctRules[2] == m_triangleSymmetricOct) ||
                        (m_triangleOctRules[3] == m_triangleSymmetricOct);
      }
    }

    if (triSymmN > 1) {
      std::cerr << m_className << "::UpdatePeriodicity:\n"
                << "    Only one triangle symmetry allowed; reset.\n";
      m_symmetries.Deactivate(Symmetry::Type::Triangle);
      m_symmetries.Deactivate(Symmetry::Type::Mirror);
      m_symmetries.Deactivate(Symmetry::Type::MirrorFinite);
      m_triangleSymmetricOct = 0;
    }
  }

  if (m_symmetries.Has(Symmetry::Type::Triangle)) {
    const bool octPos =
        (m_triangleSymmetricOct == 1 || m_triangleSymmetricOct == 2 ||
         m_triangleSymmetricOct == 7 || m_triangleSymmetricOct == 8);

    if (m_symmetries.Has(Symmetry::Type::TriangleXY)) {
      if (octPos) {
        m_bbMin[0] = -m_bbMax[0];
        m_bbMin[1] = -m_bbMax[1];
      } else {
        m_bbMax[0] = -m_bbMin[0];
        m_bbMax[1] = -m_bbMin[1];
      }
    }

    if constexpr (N == 3) {
      if (m_symmetries.Has(Symmetry::Type::TriangleXZ)) {
        if (octPos) {
          m_bbMin[0] = -m_bbMax[0];
          m_bbMin[2] = -m_bbMax[2];
        } else {
          m_bbMax[0] = -m_bbMin[0];
          m_bbMax[2] = -m_bbMin[2];
        }
      }

      if (m_symmetries.Has(Symmetry::Type::TriangleYZ)) {
        if (octPos) {
          m_bbMin[1] = -m_bbMax[1];
          m_bbMin[2] = -m_bbMax[2];
        } else {
          m_bbMax[1] = -m_bbMin[1];
          m_bbMax[2] = -m_bbMin[2];
        }
      }
    }
  }
}

template <size_t N>
void ComponentTcadBase<N>::Cleanup() {
  // Vertices
  m_vertices.clear();
  // Elements
  m_elements.clear();
  // Regions
  m_regions.clear();
  // Potential and electric field.
  m_epot.clear();
  m_efield.clear();
  // Weighting potential and field.
  m_wpot.clear();
  m_wfield.clear();
  m_wshift.clear();
  m_dwf.clear();
  m_dwp.clear();
  m_dwtp.clear();
  m_dwtf.clear();

  // Other data.
  m_eVelocity.clear();
  m_hVelocity.clear();
  m_eMobility.clear();
  m_hMobility.clear();
  m_eAlpha.clear();
  m_hAlpha.clear();
  m_eLifetime.clear();
  m_hLifetime.clear();
  m_defects.clear();
  m_eEta.clear();
  m_hEta.clear();
}

template <size_t N>
void ComponentTcadBase<N>::MapCoordinates(std::array<double, N>& x,
                                          std::array<bool, N>& mirr) const {
  mirr.fill(false);
  for (size_t i = 0; i < N; ++i) {
    const Symmetry sym = m_symmetries.GetSymmetries(i);
    const double cellsx = m_bbMax[i] - m_bbMin[i];

    if (sym.Has(Symmetry::Type::Periodic)) {
      // Infinite periodic wrapping
      x[i] = m_bbMin[i] + fmod(x[i] - m_bbMin[i], cellsx);
      if (x[i] < m_bbMin[i]) x[i] += cellsx;

    } else if (sym.Has(Symmetry::Type::Mirror)) {
      // Infinite mirror periodicity
      double xNew = m_bbMin[i] + fmod(x[i] - m_bbMin[i], cellsx);
      if (xNew < m_bbMin[i]) xNew += cellsx;
      const int nx = int(floor(0.5 + (xNew - x[i]) / cellsx));
      if (nx != 2 * (nx / 2)) {
        xNew = m_bbMin[i] + m_bbMax[i] - xNew;
        mirr[i] = true;
      }
      x[i] = xNew;

    } else if (sym.Has(Symmetry::Type::MirrorFinite)) {
      int nOps = std::max(0, m_symmetries.GetNOps(i));
      double xi = x[i];

      while (nOps > 0 && (xi < m_bbMin[i] || xi > m_bbMax[i])) {
        if (xi < m_bbMin[i]) {
          xi = 2. * m_bbMin[i] - xi;
          mirr[i] = !mirr[i];
        } else if (xi > m_bbMax[i]) {
          xi = 2. * m_bbMax[i] - xi;
          mirr[i] = !mirr[i];
        }
        --nOps;
      }
      x[i] = xi;
    }
  }
  // Triangle symmetry
  if ((N >= 2) && m_symmetries.Has(Symmetry::Type::Triangle)) {
    size_t a = 0, b = 1;
    if (m_symmetries.Has(Symmetry::Type::TriangleXY)) {
      a = 0;
      b = 1;
    } else if (m_symmetries.Has(Symmetry::Type::TriangleXZ)) {
      if (N < 3) return;
      a = 0;
      b = 2;
    } else {
      if (N < 3) return;
      a = 1;
      b = 2;
    }

    const double shiftFromDiagonal = 0.25e-4;
    if (std::abs(x[a] - x[b]) < Small) x[a] += shiftFromDiagonal;

    const int oct = m_triangleSymmetricOct;  // 1..8
    const double aa = std::abs(x[a]), bb = std::abs(x[b]);

    const bool triSwap = m_outsideCone ? (aa <= bb) : (aa >= bb);

    const double sa = (oct >= 3 && oct <= 5) ? -1. : +1.;
    const double sb = (oct >= 4 && oct <= 7) ? -1. : +1.;

    if (triSwap) {
      x[a] = sa * bb;
      x[b] = sb * aa;
    } else {
      x[a] = sa * aa;
      x[b] = sb * bb;
    }
  }
}

template <size_t N>
size_t ComponentTcadBase<N>::FindRegion(const std::string& name) const {
  const auto nRegions = m_regions.size();
  for (size_t j = 0; j < nRegions; ++j) {
    if (name == m_regions[j].name) return j;
  }
  return m_regions.size();
}

template <size_t N>
void ComponentTcadBase<N>::UpdateAttachment() {
  if (!m_useLifetimeMap && !m_useTrapOccMap) {
    m_eEta.clear();
    m_hEta.clear();
  }
  if (m_vertices.empty()) return;
  const size_t nVertices = m_vertices.size();
  m_eEta.assign(nVertices, 0.);
  m_hEta.assign(nVertices, 0.);
  if (m_useLifetimeMap) ComputeEtaFromLifetime();
  if (m_useTrapOccMap) ComputeEtaFromTraps();
}

template <size_t N>
void ComponentTcadBase<N>::ComputeEtaFromLifetime() {
  if (m_vertices.empty()) return;
  const size_t nVertices = m_vertices.size();
  if (!m_eLifetime.empty()) {
    for (size_t i = 0; i < nVertices; ++i) {
      if (m_eLifetime[i] > 0.) {
        m_eEta[i] = -1. / m_eLifetime[i];
      }
    }
  }
  if (!m_hLifetime.empty()) {
    for (size_t i = 0; i < nVertices; ++i) {
      if (m_hLifetime[i] > 0.) {
        m_hEta[i] = -1. / m_hLifetime[i];
      }
    }
  }
}

template <size_t N>
void ComponentTcadBase<N>::ComputeEtaFromTraps() {
  if (m_vertices.empty()) return;
  const size_t nVertices = m_vertices.size();

  const size_t nDefects = m_defects.size();
  for (size_t i = 0; i < nDefects; ++i) {
    const auto& defect = m_defects[i];
    if (defect.conc < 0.) continue;
    const double re = defect.conc * defect.xsece;
    const double rh = defect.conc * defect.xsech;
    if (defect.type == DefectType::Donor) {
      for (size_t j = 0; j < nVertices; ++j) {
        // Get the occupation probability.
        const double f = m_defectOcc[j][i];
        if (re > 0.) m_eEta[j] += re * f;
        if (rh > 0.) m_hEta[j] += rh * (1. - f);
      }
    } else if (defect.type == DefectType::Acceptor) {
      for (size_t j = 0; j < nVertices; ++j) {
        const double f = m_defectOcc[j][i];
        if (re > 0.) m_eEta[j] += re * (1. - f);
        if (rh > 0.) m_hEta[j] += rh * f;
      }
    }
  }
}

template <size_t N>
void ComponentTcadBase<N>::CopyWeightingPotential(
    const std::string& label, const std::string& labelSource, const double x,
    const double y, const double z, const double alpha, const double beta,
    const double gamma) {
  // Already exists?
  if (m_wpot.count(label) > 0) {
    std::cout << m_className << "::CopyWeightingPotential:\n"
              << "    Electrode " << label << " exists already.\n";
    return;
  }
  if (m_wfieldCopies.count(label) > 0) {
    std::cout << m_className << "::CopyWeightingPotential:\n"
              << "    A copy named " << label << " exists already.\n";
    return;
  }
  if (m_wpot.count(labelSource) == 0) {
    std::cout << m_className << "::CopyWeightingPotential:\n"
              << "    Source electrode " << labelSource << " does not exist.\n";
    return;
  }

  WeightingFieldCopy w;
  w.source = labelSource;

  const double ca = std::cos(-alpha), sa = std::sin(-alpha);
  const double cb = std::cos(-beta), sb = std::sin(-beta);
  const double cg = std::cos(-gamma), sg = std::sin(-gamma);

  // Rx
  double Rx[3][3] = {{1, 0, 0}, {0, ca, -sa}, {0, sa, ca}};
  // Ry
  double Ry[3][3] = {{cb, 0, sb}, {0, 1, 0}, {-sb, 0, cb}};
  // Rz
  double Rz[3][3] = {{cg, -sg, 0}, {sg, cg, 0}, {0, 0, 1}};

  double temp[3][3];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      temp[r][c] =
          Rx[r][0] * Ry[0][c] + Rx[r][1] * Ry[1][c] + Rx[r][2] * Ry[2][c];
    }
  }

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      w.rot[r][c] =
          temp[r][0] * Rz[0][c] + temp[r][1] * Rz[1][c] + temp[r][2] * Rz[2][c];
    }
  }

  w.trans[0] = -x;
  w.trans[1] = -y;
  w.trans[2] = -z;

  m_wfieldCopies[label] = w;

  std::cout << m_className << "::CopyWeightingPotential:\n"
            << "    Copy named " << label << " of weighting potential "
            << labelSource << " made.\n";
}

template <size_t N>
const std::vector<double>& ComponentTcadBase<N>::DelayedSignalTimes(
    const std::string& label) {
  const auto itp = m_dwtp.find(label);
  const auto itf = m_dwtf.find(label);
  if (itp == m_dwtp.end() && itf == m_dwtf.end()) {
    // Neither potential nor field are defined for this electrode.
    static const std::vector<double> emptyVector;
    return emptyVector;
  }
  if (itp != m_dwtp.end()) {
    return m_dwtp[label];
  }
  return m_dwtf[label];
}

template <size_t N>
std::vector<std::string> ComponentTcadBase<N>::GetElectrodeLabels() const {
  std::vector<std::string> labels;
  labels.reserve(m_wpot.size() + m_wfieldCopies.size());
  for (const auto& kv : m_wpot) labels.push_back(kv.first);
  for (const auto& kv : m_wfieldCopies) labels.push_back(kv.first);
  return labels;
}
template class ComponentTcadBase<2>;
template class ComponentTcadBase<3>;
}  // namespace Garfield

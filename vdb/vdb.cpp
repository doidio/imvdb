#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/FastSweeping.h>


namespace py = pybind11;
using namespace openvdb::OPENVDB_VERSION_NAME;

using GridT = FloatGrid;

class pyGrid {
public:
    explicit pyGrid(GridT::ValueType background) {
        this->ptr = GridT::create(background);
    }

    explicit pyGrid(GridT::Ptr ptr) {
        this->ptr = std::move(ptr);
    }

    GridT::Ptr ptr;
};

pyGrid py_from_array(py::array &array, const py::list &origin, const py::list &spacing, GridT::ValueType background,
                     GridT::ValueType tolerance) {
    if (!array.dtype().is(py::dtype::of<GridT::ValueType>())) {
        throw TypeError{};
    }

    Coord shape = {Int32(array.shape(0)) - 1,
                   Int32(array.shape(1)) - 1,
                   Int32(array.shape(2)) - 1};
    tools::Dense<GridT::ValueType> dense(CoordBBox({}, shape),
                                         static_cast<GridT::ValueType *>(array.mutable_data()));
    pyGrid grid(background);
    tools::copyFromDense(dense, *grid.ptr, tolerance);

    Vec3d originVec;
    for (py::ssize_t _ = 0; _ < origin.size(); ++_) {
        originVec[_] = origin[_].cast<Vec3d::ValueType>();
    }
    grid.ptr->insertMeta("origin", Vec3DMetadata(originVec));

    Vec3d spacingVec;
    for (py::ssize_t _ = 0; _ < spacing.size(); ++_) {
        spacingVec[_] = spacing[_].cast<Vec3d::ValueType>();
    }
    grid.ptr->insertMeta("spacing", Vec3DMetadata(spacingVec));

    Vec3I shapeVec;
    for (py::ssize_t _ = 0; _ < array.ndim(); ++_) {
        shapeVec[_] = array.shape(_);
    }
    grid.ptr->insertMeta("shape", Vec3IMetadata(shapeVec));

    math::Transform::Ptr t = math::Transform::createLinearTransform();
    t->postScale(spacingVec);
    t->postTranslate(originVec);
    grid.ptr->setTransform(t);

    return grid;
}

py::array py_to_array(pyGrid &grid) {
    Vec3I shapeVec = grid.ptr->getMetadata<Vec3IMetadata>("shape")->value();
    py::array::ShapeContainer shape;
    for (py::ssize_t _ = 0; _ < Vec3I::size; ++_) {
        shape->push_back(shapeVec[_]);
    }
    py::array_t<GridT::ValueType> array(shape);

    tools::Dense<GridT::ValueType> dense(CoordBBox({}, Coord(shapeVec - 1)),
                                         static_cast<GridT::ValueType *>(array.mutable_data()));
    tools::copyToDense(*grid.ptr, dense);
    return array;
}

void py_write(const py::list &grids, const std::string &filename) {
    GridPtrVec vec;
    for (const auto &grid: grids) {
        vec.push_back(grid.cast<pyGrid>().ptr);
    }

    io::File file(filename);
    file.write(vec);
    file.close();
}

pyGrid py_fog_to_sdf(pyGrid &grid, GridT::ValueType iso_value) {
    return pyGrid(tools::fogToSdf(*grid.ptr, iso_value));
}

PYBIND11_MODULE(vdb, m) {
    m.doc() = "openvdb python bindings";

    initialize();

    py::class_<pyGrid>(m, "Grid")
            .def(py::init<GridT::ValueType>())
            .def("__repr__", [](const pyGrid &grid) {
                std::ostringstream os;
                grid.ptr->print(os, 4);
                return os.str();
            })
            .def_property(
                    "name",
                    [](const pyGrid &grid) { return grid.ptr->getName(); },
                    [](const pyGrid &grid, const std::string &value) { grid.ptr->setName(value); })
            .def_property(
                    "creator",
                    [](const pyGrid &grid) { return grid.ptr->getCreator(); },
                    [](const pyGrid &grid, const std::string &value) { grid.ptr->setCreator(value); })
            .def_property(
                    "grid_class",
                    [](const pyGrid &grid) { return grid.ptr->gridClassToString(grid.ptr->getGridClass()); },
                    [](const pyGrid &grid, const std::string &value) {
                        for (int _ = GRID_UNKNOWN; _ < NUM_GRID_CLASSES; ++_) {
                            if (grid.ptr->gridClassToString(GridClass(_)) == value) {
                                grid.ptr->setGridClass(GridClass(_));
                            }
                        }
                    })
            .def_property_readonly(
                    "background",
                    [](const pyGrid &grid) -> GridT::ValueType { return grid.ptr->background(); })
            .def_property_readonly("metadata", [](const pyGrid &grid) -> py::dict {
                py::dict metadata;
                for (auto _ = grid.ptr->beginMeta(); _ != grid.ptr->endMeta(); ++_) {
                    metadata[_->first.c_str()] = _->second->str();
                }
                return metadata;
            });

    py::enum_<GridClass>(m, "GridClass")
            .value("GRID_UNKNOWN", GridClass::GRID_UNKNOWN)
            .value("GRID_LEVEL_SET", GridClass::GRID_LEVEL_SET)
            .value("GRID_FOG_VOLUME", GridClass::GRID_FOG_VOLUME)
            .value("GRID_STAGGERED", GridClass::GRID_STAGGERED);

    m.def("from_array", &py_from_array);
    m.def("to_array", &py_to_array);
    m.def("write", &py_write);
    m.def("fog_to_sdf", &py_fog_to_sdf);
}
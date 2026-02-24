#include "coxdev.h"

PYBIND11_MODULE(coxc, m) {
  m.doc() = "Cumsum implementations";
  m.def("forward_cumsum", [](const Eigen::Ref<const Eigen::VectorXd>& sequence, Eigen::Ref<Eigen::VectorXd> output) {
      forward_cumsum(sequence, output);
  }, "Cumsum a vector");
  m.def("reverse_cumsums", [](const Eigen::Ref<const Eigen::VectorXd>& sequence, Eigen::Ref<Eigen::VectorXd> event_buffer, Eigen::Ref<Eigen::VectorXd> start_buffer, const Eigen::Ref<const Eigen::VectorXi>& event_order, const Eigen::Ref<const Eigen::VectorXi>& start_order, bool do_event, bool do_start) {
      reverse_cumsums(sequence, event_buffer, start_buffer, event_order, start_order, do_event, do_start);
  }, "Reversed cumsum a vector");
  m.def("to_native_from_event", [](Eigen::Ref<Eigen::VectorXd> arg, const Eigen::Ref<const Eigen::VectorXi>& event_order, Eigen::Ref<Eigen::VectorXd> reorder_buffer) {
      to_native_from_event(arg, event_order, reorder_buffer);
  }, "To Native from event");
  m.def("to_event_from_native", [](const Eigen::Ref<const Eigen::VectorXd>& arg, const Eigen::Ref<const Eigen::VectorXi>& event_order, Eigen::Ref<Eigen::VectorXd> reorder_buffer) {
      to_event_from_native(arg, event_order, reorder_buffer);
  }, "To Event from native");
  m.def("forward_prework", [](const Eigen::Ref<const Eigen::VectorXi>& status,
                              const Eigen::Ref<const Eigen::VectorXd>& w_avg,
                              const Eigen::Ref<const Eigen::VectorXd>& scaling,
                              const Eigen::Ref<const Eigen::VectorXd>& risk_sums,
                              int i, int j,
                              Eigen::Ref<Eigen::VectorXd> moment_buffer,
                              const Eigen::Ref<const Eigen::VectorXd>& arg,
                              bool use_w_avg) {
      forward_prework(status, w_avg, scaling, risk_sums, i, j, moment_buffer, arg, use_w_avg);
  }, "Cumsums of scaled and weighted quantities");
  
  m.def("compute_sat_loglik", [](const Eigen::Ref<const Eigen::VectorXi>& first,
                                 const Eigen::Ref<const Eigen::VectorXi>& last,
                                 const Eigen::Ref<const Eigen::VectorXd>& weight,
                                 const Eigen::Ref<const Eigen::VectorXi>& event_order,
                                 const Eigen::Ref<const Eigen::VectorXi>& status,
                                 const Eigen::Ref<const Eigen::VectorXd>& scaling,
                                 Eigen::Ref<Eigen::VectorXd> W_status,
				 bool efron) {
    return compute_sat_loglik(first, last, weight, event_order, status, scaling, W_status, efron);
  }, "Compute saturated log likelihood");
  
  m.def("cox_dev", [](const Eigen::Ref<const Eigen::VectorXd>& eta,
                      const Eigen::Ref<const Eigen::VectorXd>& sample_weight,
                      const Eigen::Ref<const Eigen::VectorXd>& exp_w,
                      const Eigen::Ref<const Eigen::VectorXi>& event_order,   
                      const Eigen::Ref<const Eigen::VectorXi>& start_order,
                      const Eigen::Ref<const Eigen::VectorXi>& status,
                      const Eigen::Ref<const Eigen::VectorXi>& first,
                      const Eigen::Ref<const Eigen::VectorXi>& last,
                      const Eigen::Ref<const Eigen::VectorXd>& scaling,
                      const Eigen::Ref<const Eigen::VectorXi>& event_map,
                      const Eigen::Ref<const Eigen::VectorXi>& start_map,
                      double loglik_sat,
                      Eigen::Ref<Eigen::VectorXd> T_1_term,
                      Eigen::Ref<Eigen::VectorXd> T_2_term,
                      Eigen::Ref<Eigen::VectorXd> grad_buffer,
                      Eigen::Ref<Eigen::VectorXd> diag_hessian_buffer,
                      Eigen::Ref<Eigen::VectorXd> diag_part_buffer,
                      Eigen::Ref<Eigen::VectorXd> w_avg_buffer,
                      py::list event_reorder_buffers,
                      py::list risk_sum_buffers,
                      py::list forward_cumsum_buffers,
                      Eigen::Ref<Eigen::VectorXd> forward_scratch_buffer,
                      py::list reverse_cumsum_buffers,
                      bool have_start_times,
                      bool efron) {
      CoxContext ctx = {event_order, start_order, status, first, last, scaling, event_map, start_map, have_start_times, efron};
      
      return cox_dev(eta, sample_weight, exp_w, ctx, loglik_sat,
                     T_1_term, T_2_term, grad_buffer, diag_hessian_buffer, diag_part_buffer, w_avg_buffer,
                     event_reorder_buffers[0].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     event_reorder_buffers[1].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     event_reorder_buffers[2].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     risk_sum_buffers[0].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     forward_cumsum_buffers[0].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     forward_cumsum_buffers[1].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     forward_cumsum_buffers[2].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     forward_cumsum_buffers[3].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     forward_cumsum_buffers[4].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     forward_scratch_buffer,
                     reverse_cumsum_buffers[0].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     reverse_cumsum_buffers[1].cast<Eigen::Ref<Eigen::VectorXd>>());
  });

  m.def("hessian_matvec", [](const Eigen::Ref<const Eigen::VectorXd>& arg,
                             const Eigen::Ref<const Eigen::VectorXd>& eta, 
                             const Eigen::Ref<const Eigen::VectorXd>& sample_weight, 
                             Eigen::Ref<Eigen::VectorXd> risk_sums,
                             Eigen::Ref<Eigen::VectorXd> diag_part,
                             Eigen::Ref<Eigen::VectorXd> w_avg,
                             Eigen::Ref<Eigen::VectorXd> exp_w,
                             Eigen::Ref<Eigen::VectorXd> event_cumsum_orig, 
                             Eigen::Ref<Eigen::VectorXd> start_cumsum_orig, 
                             const Eigen::Ref<const Eigen::VectorXi>& event_order,   
                             const Eigen::Ref<const Eigen::VectorXi>& start_order,
                             const Eigen::Ref<const Eigen::VectorXi>& status,
                             const Eigen::Ref<const Eigen::VectorXi>& first,
                             const Eigen::Ref<const Eigen::VectorXi>& last,
                             const Eigen::Ref<const Eigen::VectorXd>& scaling,
                             const Eigen::Ref<const Eigen::VectorXi>& event_map,
                             const Eigen::Ref<const Eigen::VectorXi>& start_map,
                             py::list risk_sum_buffers,
                             py::list forward_cumsum_buffers,
                             Eigen::Ref<Eigen::VectorXd> forward_scratch_buffer,
                             py::list reverse_cumsum_buffers,
                             Eigen::Ref<Eigen::VectorXd> hess_matvec_buffer,
                             bool have_start_times,
                             bool efron) {
      CoxContext ctx = {event_order, start_order, status, first, last, scaling, event_map, start_map, have_start_times, efron};
      
      hessian_matvec(arg, ctx, risk_sums, diag_part, w_avg, exp_w,
                     risk_sum_buffers[1].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     forward_cumsum_buffers[0].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     forward_cumsum_buffers[1].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     forward_scratch_buffer,
                     reverse_cumsum_buffers[2].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     reverse_cumsum_buffers[3].cast<Eigen::Ref<Eigen::VectorXd>>(),
                     hess_matvec_buffer);
  });

  m.def("c_preprocess", [](const Eigen::Ref<const Eigen::VectorXd>& start,
                           const Eigen::Ref<const Eigen::VectorXd>& event,
                           const Eigen::Ref<const Eigen::VectorXi>& status,
                           const Eigen::Ref<const Eigen::VectorXd>& weight) {
      Eigen::VectorXd _start, _event, _scaling;
      Eigen::VectorXi _status, _first, _last, _start_map, _event_map, event_order, start_order;
      setup_preprocess(start, event, status, weight, _start, _event, _status, _first, _last, _scaling, _start_map, _event_map, event_order, start_order);
      
      py::dict preproc;
      preproc["start"] = _start;
      preproc["event"] = _event;
      preproc["first"] = _first;
      preproc["last"] = _last;
      preproc["scaling"] = _scaling;
      preproc["start_map"] = _start_map;
      preproc["event_map"] = _event_map;
      preproc["status"] = _status;
      return std::make_tuple(preproc, event_order, start_order);
  }, "C Preprocessing");

  py::class_<CoxDeviance, std::shared_ptr<CoxDeviance>>(m, "CoxDeviance")
      .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXi&, const Eigen::VectorXd&, bool>())
      .def("compute_deviance", &CoxDeviance::compute_deviance)
      .def("compute_hessian_matvec", [](CoxDeviance &self, const Eigen::Ref<const Eigen::VectorXd>& arg) {
          Eigen::VectorXd out(arg.size());
          self.compute_hessian_matvec(arg, out);
          return out;
      })
      .def_property_readonly("gradient", &CoxDeviance::get_gradient, py::return_value_policy::reference_internal)
      .def_property_readonly("diag_hessian", &CoxDeviance::get_diag_hessian, py::return_value_policy::reference_internal)
      .def_property_readonly("linear_predictor", &CoxDeviance::get_linear_predictor, py::return_value_policy::reference_internal)
      .def_property_readonly("sample_weight", &CoxDeviance::get_sample_weight, py::return_value_policy::reference_internal)
      .def_property_readonly("loglik_sat", &CoxDeviance::get_loglik_sat)
      .def_property_readonly("event_order", &CoxDeviance::get_event_order, py::return_value_policy::reference_internal)
      .def_property_readonly("start_order", &CoxDeviance::get_start_order, py::return_value_policy::reference_internal)
      .def_property_readonly("first", &CoxDeviance::get_first, py::return_value_policy::reference_internal)
      .def_property_readonly("last", &CoxDeviance::get_last, py::return_value_policy::reference_internal)
      .def_property_readonly("start_map", &CoxDeviance::get_start_map, py::return_value_policy::reference_internal)
      .def_property_readonly("event_map", &CoxDeviance::get_event_map, py::return_value_policy::reference_internal)
      .def_property_readonly("scaling", &CoxDeviance::get_scaling, py::return_value_policy::reference_internal)
      .def_property_readonly("status", &CoxDeviance::get_status, py::return_value_policy::reference_internal)
      .def_property_readonly("event", &CoxDeviance::get_event, py::return_value_policy::reference_internal)
      .def_property_readonly("start", &CoxDeviance::get_start, py::return_value_policy::reference_internal);

  py::class_<StratifiedCoxDeviance>(m, "StratifiedCoxDeviance")
      .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXd&, bool>())
      .def("compute_deviance", &StratifiedCoxDeviance::compute_deviance)
      .def("compute_hessian_matvec", [](StratifiedCoxDeviance &self, const Eigen::Ref<const Eigen::VectorXd>& arg) {
          Eigen::VectorXd out(arg.size());
          self.compute_hessian_matvec(arg, out);
          return out;
      })
      .def_property_readonly("gradient", &StratifiedCoxDeviance::get_gradient, py::return_value_policy::reference_internal)
      .def_property_readonly("diag_hessian", &StratifiedCoxDeviance::get_diag_hessian, py::return_value_policy::reference_internal)
      .def_property_readonly("linear_predictor", &StratifiedCoxDeviance::get_linear_predictor, py::return_value_policy::reference_internal)
      .def_property_readonly("sample_weight", &StratifiedCoxDeviance::get_sample_weight, py::return_value_policy::reference_internal)
      .def_property_readonly("loglik_sat", &StratifiedCoxDeviance::get_loglik_sat)
      .def_property_readonly("cox_devs", &StratifiedCoxDeviance::get_cox_devs, py::return_value_policy::reference_internal)
      .def_property_readonly("unique_strata", &StratifiedCoxDeviance::get_unique_strata, py::return_value_policy::reference_internal)
      .def_property_readonly("stratum_indices", &StratifiedCoxDeviance::get_stratum_indices, py::return_value_policy::reference_internal);
}
